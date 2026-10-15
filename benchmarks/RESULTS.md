# gnss_gpu Performance Benchmark Results

測定条件と再現コマンドは各節に記載する。過去の `bench_all.py` の表は3回実行の中央値。

統合時の型検査注記はPython 3.12の再帰的compiled module比較で実行コード不変を確認した。
[統合時の照合記録](optimization_merge_check_20261015.json)。

## Exact uniform inverse, coalesced state gather and reset in one kernel (2026-10-15)

**D202比、100万→50万→100万粒子の切替往復を3.419→3.055ms（10.66%短縮）、切替込み連続5epochを62.800→61.386ms（2.25%短縮）した。**
PR posterior切替は28.904→28.955ms（0.17%悪化）。全測定と悪化したround/controlも保存する。
元のFP64-RNしきい値、整数区間逆検索・逆数/high64/積の補正、left同値境界とclampで同じ粒子番号を求める。
選択・7scalar/9covarianceの全16状態copy・+0.0重みresetを1kernelへ統合し、ancestor-Aの書込み/読込みと2launchを省く。
全laneがshared ancestorを初期化して1回の無条件barrierへ到達し、端数blockも元の連続component/lane対応で共分散をcopyする。
D202のsourceN>=262144 gate、全重みの毎回GPU検査、NumPy policy/getter/CDF identity guard、cache無効化/同期を維持する。
診断CDF出力を要求した場合は全CDFを従来通り生成し、metadata容量不足では元の個別kernel処理へ戻る。小粒子の処理本体はD202と同一。
新CUDA owner/確保/PF layout変更はない。全87既存GPU命令/resourceは同一。粒子数/乱数/FP64/測位モデル/許容誤差は変更しない。

| Complete synchronized stage, N1000000 | Before ms | After ms | Round-median reduction | Median per-pair reduction | Improving rounds |
|---|---:|---:|---:|---:|---:|
| public_pool_roundtrip | 3.419 | 3.055 | 10.66% | 9.79% | 5/5 |
| public_pool_PR_five | 62.800 | 61.386 | 2.25% | 2.15% | 5/5 |
| PR_control | 5.308 | 5.302 | 0.11% | -0.29% | 2/5 |

上表は追加の5round×9repeat×3stage=135交互pair/15snapshot。5epochの45pairは全て短縮した。
その前の5round×5repeat×9stage=225pairでは、5epochは63.766→63.096ms、1.05%短縮、3/5round改善と弱かったため反復数を増やして確認した。先行225pairの全結果も以下へ示す。

| Earlier complete synchronized stage, N1000000 | Before ms | After ms | Round-median reduction | Median per-pair reduction | Improving rounds |
|---|---:|---:|---:|---:|---:|
| public_cdf_validate | 0.667 | 0.658 | 1.35% | -0.15% | 2/5 |
| public_cdf_decline_nan | 0.713 | 0.839 | -17.73% | -11.46% | 3/5 |
| public_cdf_decline_decrease | 0.701 | 0.750 | -7.02% | 2.04% | 4/5 |
| public_pool_roundtrip | 3.537 | 3.214 | 9.13% | 10.03% | 5/5 |
| public_pool_posterior | 28.904 | 28.955 | -0.17% | -0.37% | 4/5 |
| public_pool_PR_five | 63.766 | 63.096 | 1.05% | 1.92% | 3/5 |
| public_states | 51.528 | 47.495 | 7.83% | -0.19% | 1/5 |
| public_restore | 22.491 | 21.495 | 4.43% | 1.59% | 3/5 |
| PR_control | 4.056 | 4.071 | -0.39% | 0.12% | 4/5 |

36条件（2048/65536/100000/1000000×9stage）の3round×3repeat、100万粒子の5round×5repeat×9stage=225交互pair/45全状態snapshot、初期75pair/15snapshotを保存。
計測セッションのpool factoryが自己参照を作り、close後もGC待ちになることをCPUの弱参照probeで再現した。計測外のclose末尾でpool/peerを解放する2代入だけを追加し、GC無効でも即時解放することを確認した。
Session.close以外の全AST、setup/validator/init/prepare/operation/timer/workerと全CUDA source/PEは同一。FP照合recordは同一source/PE/検証関数でfresh実行した6,904件を保持し、元のhelper/検証JSONのhashとcloseだけの再構成patchを残した。
修正前の36条件/225pair/6peakもJSONに保存し、終了処理を揃えた再測定を主結果に使う。計測対象のAPI演算は変更していない。
同じ論理CPUの既知の2workerで直列測定。非独占hostで、初期化/復元/input/IPC/hashは計測外、検査/正規化/転送/GPU処理/同期は計測内。
未変更の小粒子やCDF検証・全状態get/set・PR単体の変動を改善へ帰属せず、全粒子数・全重み条件へ同率の改善を一般化しない。
全6,904候補recordをfresh実行し、hash確認済みD202の6,817参照+87追加baseline fresh recordと一致（6,824出力/80同じ例外）。
追加87件は3source×13target×2variantの78件、後続predict/KF/resampleとscratch再利用3件、診断用全CDF bit/全状態6件。
最終版の回帰11,375pass（CPUの解放回帰1件を含む）。追加GPU87件のmemcheck/racecheck/synccheckはすべて0error/leak/hazard。6件のPython/NumPy traced peakと全状態snapshotも確認。
tracemallocはC++/GPU確保を含まない。長時間実データの測位誤差曲線・実機multi-device/他archは追加評価していない。
[測定・照合・検査](pf_fused_uniform_gpu_20261015.json)、[コード・再構成patch](pf_fused_uniform_device_code_20261013.json)、[完了監査](pf_fused_uniform_completion_check_20261015.json)。
再現: `bench_pf_fused_uniform_compare.py --baseline <D202.pyd> --candidate <new.pyd> --baseline-dir <frozen-D202-python> --sizes 2048 65536 100000 1000000 --rounds 3 --repeats 3 --output <json>`。
交互測定は `--sizes 1000000 --rounds 5 --repeats 5 --paired`。Release/CUDA12.8/MSVC/compute7.5/C++20/no fast math。

## Exact uniform CDF inverse after full GPU weight scan (2026-10-15)

**D201比で100万→50万→100万粒子の切替往復を4.242→3.618ms（14.72%短縮）した。切替込み連続5epochは65.076→62.572ms（3.85%短縮）。**
PR posterior切替は21.421→21.892ms、約2.20%悪化。不均一重みを含む切替の短縮は今回再現できず、その結果も保持する。
D200で保留した厳密なCDF逆検索をD201の上で再評価し、100万粒子の全体への安定した効果を確認して採用した。sourceNが262144未満ならD201の元のhost/CDF/検索本体をそのまま実行する。
逐次FP64 cumsumと全ビット同じ値を表す区間表を使い、元のRNしきい値に初めて届く粒子番号を整数で求める。
整数逆数/high64と積の補正でceilingを厳密に計算し、leftの同値境界と最後のclampを維持する。粒子数/反復/乱数/測位モデル/許容誤差は変更しない。
通常はCDF全体の書込みと二分探索を省く。診断出力を要求した場合は元のCDFを全ビット生成し、逆数表が既存scratchに収まらない場合は元のCDF生成・探索へ戻る。
D201の全weight検査/4byte読戻し/NumPy policy・hook guard、元の全16状態gather/重みsetterを保つ。新CUDA owner/確保、PF layout変更はない。

| Complete synchronized stage, N1000000 | Before ms | After ms | Round-median reduction | Median per-pair reduction | Improving rounds |
|---|---:|---:|---:|---:|---:|
| public_cdf_validate | 0.557 | 0.556 | 0.18% | -1.74% | 1/5 |
| public_cdf_decline_nan | 0.594 | 0.561 | 5.60% | -0.76% | 3/5 |
| public_cdf_decline_decrease | 0.570 | 0.545 | 4.33% | -0.53% | 4/5 |
| public_pool_roundtrip | 4.242 | 3.618 | 14.72% | 14.40% | 5/5 |
| public_pool_posterior | 21.421 | 21.892 | -2.20% | 1.88% | 2/5 |
| public_pool_PR_five | 65.076 | 62.572 | 3.85% | 3.60% | 5/5 |
| public_states | 43.590 | 45.824 | -5.13% | 0.88% | 2/5 |
| public_restore | 20.227 | 20.143 | 0.41% | 0.35% | 3/5 |
| PR_control | 5.187 | 5.164 | 0.45% | -0.23% | 2/5 |

| Smaller paired condition | Stage | Before ms | After ms | Round-median reduction | Improving rounds |
|---:|---|---:|---:|---:|---:|
| 2048 | public_pool_roundtrip | 0.483 | 0.476 | 1.31% | 3/5 |
| 2048 | public_pool_posterior | 0.498 | 0.495 | 0.50% | 5/5 |
| 2048 | public_pool_PR_five | 3.375 | 3.412 | -1.10% | 2/5 |
| 65536 | public_pool_roundtrip | 1.137 | 1.129 | 0.76% | 3/5 |
| 65536 | public_pool_posterior | 2.183 | 2.308 | -5.69% | 2/5 |
| 65536 | public_pool_PR_five | 9.980 | 10.135 | -1.55% | 3/5 |
| 100000 | public_pool_roundtrip | 3.927 | 3.712 | 5.47% | 5/5 |
| 100000 | public_pool_posterior | 5.957 | 6.208 | -4.21% | 1/5 |
| 100000 | public_pool_PR_five | 14.466 | 14.596 | -0.90% | 3/5 |

5round×5repeat×9stage=225交互pair、45全状態snapshot、4粒子数×9stage=36条件の3round×3repeat、初期75pair/15snapshotを保存。
別途2048/65536/100000粒子×9stageの675交互pair/135snapshotを測定した。
適用範囲を制限する前のbatchの10万粒子ではuniform切替約7.1%/posterior切替約21.1%悪化し、2048粒子の未変更5epochも約5.0%悪化した。これらを除かず保存し、制限前後の全結果をJSONへ残す。上表の最終版の小〜中粒子数は元の処理であり、時間の変動を今回の改善に帰属しない。
同じ論理CPUの既知の2workerを使い、全GPU jobは直列、非独占hostで測定。初期化/復元/IPC/hashは計測外、検証/正規化/転送/GPU処理/同期/配列生成は計測内。
未変更controlと悪化したroundも残し、その変動を今回の改善に帰属しない。小粒子や非均一重みへ同率の改善を一般化しない。
全6,817候補recordをfresh実行し、hash確認したD201の6,799参照+18追加fresh recordと一致（6,737出力/80例外）。新旧各530unit、最終版の回帰11,287pass。
最終版では境界前後・metadata fallbackの18件をCUDA検査3種でfresh実行し、error/leak/hazardなし。制限前候補では各237件を検査済みで、最終版の全87GPU命令/resourceが同一と確認した。
cached/uncached source ESS/estimate/spreadの12独立oracle probeとCUDA検査3種は制限前候補の記録を保持し、全87命令/resource一致を確認。D201比で全86既存GPU命令/resourceを維持。
6件のPython/NumPy traced peakと全状態snapshotを確認。C++/GPU確保はtracemallocの対象外。長時間実データの測位誤差曲線・実機multi-device/他archは追加評価していない。
[測定と検証](pf_scan_inverse_gpu_20261015.json)、[コード・再構成patch](pf_scan_inverse_device_code_20261013.json)、[完了監査](pf_scan_inverse_completion_check_20261015.json)。
再現: `bench_pf_scan_inverse_compare.py --baseline <D201.pyd> --candidate <new.pyd> --baseline-dir <frozen-D201-python> --sizes 2048 65536 100000 1000000 --rounds 3 --repeats 3 --output <json>`。
交互測定は `--sizes 1000000 --rounds 5 --repeats 5 --paired`。Release/CUDA12.8/MSVC/compute7.5/C++20/no fast math。

## Full GPU finite/equal-weight scan before exact pool migration (2026-10-15)

**100万→50万→100万粒子の切替往復を12.27→4.23ms、約65.5%短縮した。**
PR posterior切替は25.35→21.64ms、約14.6%短縮、PR処理と毎epochの切替を含む連続5epochは105.08→64.39ms、約38.7%短縮。
直前D199の6source/PE/PF Python/realtime/helperを凍結し、重み全体のD2HとCPUのfinite/equality検査をGPUへ移した。

毎回すべての現在のlog-weightをGPUで検査する。FP64の指数bitでNaN/Infをrejectし、有限値はbit equality、正負のzeroだけ同値として扱う。
全threadの不一致flagをblock内・block間で整数ORするため、精度低下や加算順序変更はない。重みの証明cacheや部分sample判定は使わない。
既存source partial-C/final-reduction scratchを借り、4byteの判定結果だけD2Hしてsource stream完了を待つ。
Trueの場合、従来と同じD199の逐次FP64 CDF生成/left検索/clamp/全16状態gather/resetと最後のuniform setterを実行する。
Falseなら可視source状態/重み/ESS cacheとtargetを保ち、元のCPU読み戻し/正規化/CDF/fallbackを実行する。
scratchは次のESS/estimateの各kernelが全使用範囲を書き直す。新CUDA owner/確保とPF state-layout変更はない。
100万粒子のuniform切替往復で、Python/NumPy traced追加peakは17,001,628→756byte。初期化/復元/状態snapshotとC++/GPU確保はこのpeakの対象外。
非均一でFalseを返す場合はGPU検査の時間が加わるため、全ての粒子数・重みで同じ改善率とはしない。

元のPython getterとnative getter、元のuniform/checked CDF関数のpair、genuine state type、公開/native粒子数の一致を確認する。
NumPyのwarn/ignoreとRuntimeWarning-as-error判定を元と同じhelperへ移し、各migrationで1回評価する。
差替えgetter/CDF hook、旧版・注入backend、小粒子、厳しいNumPy policy、cold targetは従来経路を保つ。
全84GPU命令body/resource、D199 CDF/threshold/検索/gather/重みkernel、FP64、粒子数、乱数、測位モデル、許容誤差を維持する。

| Complete synchronized stage, N1000000 | Before ms | After ms | Round-median reduction | Median per-pair reduction | Improving rounds |
|---|---:|---:|---:|---:|---:|
| public_cdf_validate | 0.547 | 0.550 | -0.51% | -0.99% | 2/5 |
| public_cdf_decline_nan | 0.553 | 0.556 | -0.45% | -1.43% | 3/5 |
| public_cdf_decline_decrease | 0.573 | 0.566 | 1.19% | 0.79% | 4/5 |
| public_pool_roundtrip | 12.269 | 4.231 | 65.51% | 65.52% | 5/5 |
| public_pool_posterior | 25.354 | 21.641 | 14.64% | 12.11% | 5/5 |
| public_pool_PR_five | 105.083 | 64.386 | 38.73% | 38.69% | 5/5 |
| public_states | 51.633 | 99.353 | -92.42% | 1.26% | 3/5 |
| public_restore | 20.370 | 19.873 | 2.44% | 2.25% | 5/5 |
| PR_control | 5.036 | 5.040 | -0.09% | 0.94% | 4/5 |

同じ論理CPUの既知の2workerで5round×5repeat×9stages=225pairを1callずつ交互測定し、45round末尾の全状態snapshotがbit一致。
2048/65536/10万/100万×9stages=36条件の3round×3repeat、2round×3repeatのprobe、初期75pair/15snapshotも保存した。
pool 5epochはpredict→12sat PR→ESS→conditional B60 resample→estimate→2回切替→estimate。
初期化/復元/input/IPC/hashは計測外、検証/正規化/CDF/転送/GPU処理/同期/配列生成は計測内。非独占host測定。
未変更のCDF host検証3control/全状態getter/host restore/PR単体の変動を改善へ帰属しない。悪化したround/controlも残す。

6,799照合record=6,719出力/80同じ例外。候補は全6,799件を新しく実行した。
baselineはsource/PE/全従来helperのhash一致を確認したD199の6,587参照recordを再利用し、212追加recordを新しく実行した。
追加は90全重み検査、39getter/CDF/policy/count dispatch、3queued producer/reduction scratch再利用、および保持した80uniform粒子選択case。
正負zero混在、同値normal/subnormal/極端有限、先頭・中央・末尾の差、NaN/Inf/SNaN payloadを含む5粒子数×18入力を照合した。
元のCDF全ビットを含む6,587参照recordも完全一致。旧新各137unit通過。5handle testは旧版でAPI不在、候補でFalseと安全なdeclineを確認する。
回帰11,269pass、3,148caseのCUDA検査3種は0error/leak/hazard。6caseのPython/NumPy traced peakも全状態snapshot一致。
さらに12caseで、GPU検査直後のsource ESS（cache有/無）/estimate/position spreadを独立oracleと旧版にbit照合し、同じ12caseへCUDA検査3種を追加した。全て0error/leak/hazard。
長時間実データの測位誤差曲線と実機multi-device/非x86-64は今回追加評価していない。
poolの均一重み切替への改善であり、通常PF/raytrace/signal simulation全体への速度向上へ一般化しない。

[全測定・照合・検査・peak](pf_uniform_scan_gpu_20261015.json)、[コード/patch監査](pf_uniform_scan_device_code_20261013.json)、[完了監査](pf_uniform_scan_completion_check_20261015.json)。
再現: `bench_pf_uniform_scan_compare.py --baseline <D199.pyd> --candidate <new.pyd> --baseline-dir <frozen-D199-PF-and-realtime-python> --sizes 2048 65536 100000 1000000 --rounds 3 --repeats 3 --output <json>`。
交互確認は `--sizes 1000000 --rounds 5 --repeats 5 --paired`。Release/CUDA12.8/MSVC/compute7.5/C++20/no fast math。

## Direct inverse of exact uniform CDF spans: held (2026-10-15)

D200ではCDF全体を作らず、厳密な区間表から同じleft検索結果を整数で直接求める候補を試した。
整数除算版と、整数逆数/high64/積の厳密補正版を保存。CPU約1,051万threshold・約19万ceil divisionが一致し、旧新各80状態testが通った。
native切替20往復の測定は47.30→38.65ms、約18.3%短縮したが、通常の連続5epochでは短縮が小さく不安定だったため保留した。
この候補の全回帰/CUDA検査は実行していない。両trialのsource/PE/helper/全pair/snapshotを保存し、D199へ完全に戻してからD201のGPU全重み検査へ進んだ。
[保留した全結果](pf_uniform_inverse_gpu_20261015.json)、[候補patch/機械語](pf_uniform_inverse_device_code_20261013.json)。

## Exact sequential uniform CDF construction on GPU (2026-10-15)

**100万→50万→100万粒子の切替往復を18.71→12.13ms、約35.1%短縮した。**
PR posterior切替は26.52→24.45ms、約7.8%短縮、PR処理と毎epochの切替を含む連続5epochは134.25→104.93ms、約21.8%短縮。
直前D198の6source/PE/PF Python/realtime/helperを凍結し、有限かつ同値の重みに対するCDF生成と転送を変更した。

CPUの `np.cumsum(1/N)` の各丸め結果を厳密に再現する。正のFP64反復加算は同じ指数区間内でULP増分が一定になる。
CPUで開始index/bit pattern/整数増分の小さな区間表を作り、奇数仮数のhalfway tieと指数境界では元と同じFP64加算を1回実行する。
GPUはuint64の乗算・加算とbitcastだけでCDF全要素を再構成する。`i/N` への置換、parallel scanによる加算順序変更、FP32化はしない。
既存source CDF/target ancestor-B scratchを借り、CDF全要素のCPU生成・host検証・H2D転送を区間表の転送へ置換する。
新しいCUDA owner/buffer/確保は追加せず、元の83GPU命令body/resource、left検索/境界clamp/全16状態gather/重みkernelを保持した。
source producer完了後、target streamで転送→CDF生成→検索→gather→reset→任意診断出力を順序化し、完了同期後に戻る。

既存の全有限/同値検査、NumPy warn/ignore policy、private copyを保つ。
resident handle、source/targetの同じoptional hook、元のchecked CDF関数のpair、公開粒子数とnative粒子数の一致を確認する。
source N<65536、target作業領域不足、CPU丸めがnearest以外、in-place/cross-device/非x86-64、旧版・注入・差替えbackendは従来経路を使う。
CDFのoptional全要素診断出力もshape/dtype/writable/contiguousとhandleを変更前に検証する。
FP64、CDF全ビット、累積順序の結果、粒子選択、粒子数、乱数、測位モデル、許容誤差を維持した。

| Complete synchronized stage, N1000000 | Before ms | After ms | Round-median reduction | Median per-pair reduction | Improving rounds |
|---|---:|---:|---:|---:|---:|
| public_cdf_validate | 0.559 | 0.543 | 2.86% | -0.10% | 4/5 |
| public_cdf_decline_nan | 0.554 | 0.548 | 1.23% | 1.08% | 3/5 |
| public_cdf_decline_decrease | 0.542 | 0.547 | -1.00% | -1.90% | 1/5 |
| public_pool_roundtrip | 18.707 | 12.134 | 35.14% | 31.78% | 5/5 |
| public_pool_posterior | 26.518 | 24.454 | 7.78% | 6.40% | 5/5 |
| public_pool_PR_five | 134.254 | 104.932 | 21.84% | 21.58% | 5/5 |
| public_states | 49.717 | 99.566 | -100.26% | -1.00% | 2/5 |
| public_restore | 19.808 | 19.956 | -0.75% | -1.18% | 2/5 |
| PR_control | 4.953 | 4.994 | -0.83% | 0.36% | 2/5 |

同じ論理CPUの既知の2workerで5round×5repeat×9stages=225pairを1callずつ交互測定し、45round末尾の全状態snapshotがbit一致。
2048/65536/10万/100万×9stages=36条件の3round×3repeatと2round×3repeatのprobeを保存した。
100万粒子の切替往復について、probeは20.55→12.40ms、3round batchは19.19→26.46ms（改善1/3round）で、batchの中央値は悪化した。
1call交互の独立確認では5/5roundで短縮したため、上の改善率には交互測定を使う。測定方式間の差も保存し、すべての実行条件で同じ改善率とは主張しない。
pool 5epochはpredict→12sat PR→ESS→conditional B60 resample→estimate→2回切替→estimate。
初期化/復元/input/IPC/hashは計測外、検証/正規化/CDF/転送/GPU処理/同期/配列生成は計測内。非独占host測定。
CDF host検証3control、全状態getter、host restore、PR単体のコードは変えておらず、その変動を改善に帰属しない。悪化したround/controlも保存。

6,587照合record=6,513出力/74同じ例外。D198の6,305recordを再照合し、282recordを追加した。
17粒子数×4target×3状態の204caseでGPUのCDF全要素、全16状態、source状態/重み/ESSを照合。
同じ2の冪の前後とseed固定randomの47粒子数でもCDF全ビットを照合した。
24dispatch/count case、4CPU丸めmode、3queued producer/CDFとancestor-B scratch再利用/次predictを検査した。
旧新各232+63test通過。後半の13caseは候補の新APIに対するhandle/shape/dtype/readonly/stride/in-place/未整列出力契約を検証し、旧版はAPI不在を確認する。
CPU数理試作では10,167通りのCDF全要素がbit一致。これはGPU実測とは別の証拠として保存した。
回帰11,052pass、2,931caseのCUDA検査3種は0error/leak/hazard。6caseのPython/NumPy traced peakも全状態snapshot一致。
長時間実データの測位誤差曲線、実機multi-device/非x86-64は今回追加評価していない。
均一重みでpoolを切り替える処理への効果であり、通常PF/raytrace/signal simulation全体の向上へ一般化しない。

[全測定・照合・検査・peak](pf_uniform_cdf_gpu_20261015.json)、[コード/patch監査](pf_uniform_cdf_device_code_20261013.json)、[完了監査](pf_uniform_cdf_completion_check_20261015.json)。

再現: `bench_pf_uniform_cdf_compare.py --baseline <D198.pyd> --candidate <new.pyd> --baseline-dir <frozen-D198-PF-and-realtime-python> --sizes 2048 65536 100000 1000000 --rounds 3 --repeats 3 --output <json>`。
交互確認は `--sizes 1000000 --rounds 5 --repeats 5 --paired`。Release/CUDA12.8/MSVC/compute7.5/C++20/no fast math。

## Exact SIMD host CDF validation for particle-pool migration (2026-10-15)

**100万→50万→100万粒子の切替往復を18.75→17.92ms、約4.5%短縮した。**
PR posterior切替は27.19→26.01ms、約4.3%短縮、PR処理と毎epochの切替を含む連続5epochは139.47→133.68ms、約4.1%短縮。
直前D197の6source/PE/PF Python/realtime/helperを凍結し、C++ bindingのCDF検証だけを比較した。

CPUで各要素を逐次検査していた有限性/単調性確認を、x86-64かつN>=65536でCPU/OSがAVX2対応なら8要素ずつ、未対応x86-64ではSSE2で4要素ずつ実行する。
Inf/quiet NaN/signaling NaNは整数の指数bit maskで検出し、NaNへの浮動小数比較を避ける。
隣接要素のFP64 `<` 比較を全組に適用し、先頭・blockの境界・AVX2の0〜7要素/SSE2の0〜3要素の末尾も検査する。
signed zero/denorm/同値plateau/極端有限値、正負のNaN payloadにも元と同じaccept/declineと粒子選択を確認した。
未整列loadは有効範囲内。小粒子とx86-64以外は元のscalar loop。SSE2 fallbackのbodyは実行済みtrialと完全一致。
検証条件を省かず、handle/shape checksの後、native関数呼出し・state/scratch変更・stream同期・GPU転送の前に全検証を終える。
不正CDFは元と同じく変更前にFalseを返す。AVX2はCPU/OSXSAVE/AVX/XCR0対応を確認し、別noinline関数に限定する。全体のbaseline ISAを上げず、小粒子/非x86-64とSSE2 fallbackを保つ。新API/array cache/owner/確保は追加しない。
実PEのhost命令も監査し、CPUID→OSXSAVE/AVX→XCR0→AVX2 bitの検査、条件付き呼出し、dispatcher/SSE2/scalar範囲にAVX/YMM命令がないことを確認した。
FP64、CDF値/累積順序/left検索/境界clamp/全16状態/重み、粒子数、乱数、測位モデル、許容誤差を維持する。
CU/header/Python/runtime/config/realtime、全83GPU SASS/resources、state layout、転送量と同期は同じ。

| Complete synchronized stage, N1000000 | Before ms | After ms | Round-median reduction | Median per-pair reduction | Improving rounds |
|---|---:|---:|---:|---:|---:|
| public_cdf_validate | 1.163 | 0.539 | 53.68% | 53.97% | 5/5 |
| public_cdf_decline_nan | 1.159 | 0.541 | 53.27% | 53.57% | 5/5 |
| public_cdf_decline_decrease | 1.181 | 0.547 | 53.69% | 53.68% | 5/5 |
| public_pool_roundtrip | 18.755 | 17.916 | 4.48% | 4.25% | 5/5 |
| public_pool_posterior | 27.194 | 26.012 | 4.35% | 3.24% | 3/5 |
| public_pool_PR_five | 139.468 | 133.683 | 4.15% | 3.83% | 4/5 |
| public_states | 41.652 | 41.770 | -0.28% | -0.94% | 2/5 |
| public_restore | 19.855 | 20.062 | -1.04% | -0.62% | 1/5 |
| PR_control | 5.846 | 5.846 | -0.01% | -0.67% | 2/5 |

同じ論理CPUに置いた既知の両workerで5round×5repeat×9stages=225pairを1callずつ交互測定。
45round末尾のsource/target全状態/重み/返却snapshotがbit一致。
2048/65536/10万/100万×9stages=36条件の3round×3repeatと、2round×3repeatのprobeも全sample/対照を保存。
CDF検証の3isolated controlは同一source/target handleで、CDF検証後のnative入口でFalseを返し、gather/transferを行わない。
有効単調CDF・末尾NaN・末尾降下を測る。計測枠内のstream syncは両方同じ。
pool 5epochはpredict→12sat PR→ESS→conditional B60 resample→estimate→2回切替→estimate。
初期化/復元/input/IPC/hashは計測外、検証/正規化/CDF/transfer/GPU処理/同期/配列生成は計測内。非独占host測定。
全状態getter/host restore/PR単体は変更箇所を通らず、その変動を本改善へ帰属しない。悪化したround/controlも保持。

6,305照合record=6,234出力/71同じ例外。D197の3,785recordを再照合し、CDFの2,520recordを追加した。
1/2/3/4/5/31/32/33/65535/65536/65537/65538/65539/65540/65541/65542/65543/100万、28入力×5forms。
8種類のSIMD末尾、signed zero/subnormal/極端有限/plateau/left ties/raw64bit、先頭・block境界・末尾のNaN/Inf/降下を検査。
contiguous/8byte offset/1byte unaligned/stride/readonlyを比較し、decline時のtarget全16状態/重み/ESSと入力bit、source全状態を保つ。
旧新とも追加2,520test通過。回帰10,757pass。
新しい検証ケースと実pool warm/cold/decline/queued scratch/handle/coercionを含む2,636caseにCUDA検査3種を実行し0error/leak/hazard。
初期SSE2だけの候補は5,745record一致・旧新1,960test通過したが、全体の改善が約2%に留まった。
そのsource/PE/helper/全測定を保存し、CPU/OS対応AVX2を追加、未対応時のSSE2 bodyを維持して全検証と全測定を再実行した。
Python/NumPy traced peakの6caseも全状態snapshot一致。1MiB allocation追跡を確認し、C++/GPU確保は対象外。
長時間実データの測位誤差曲線と実機multi-device/non-AVX2/non-x86は今回追加評価していない。
poolのCDF検証を通る処理への改善。通常PF/raytrace/signal simulation全体の速度向上へ一般化しない。

[全測定・照合・検査・peak](pf_cdf_validate_gpu_20261015.json)、[コード/patch監査](pf_cdf_validate_device_code_20261013.json)、[完了監査](pf_cdf_validate_completion_check_20261015.json)。

再現: `bench_pf_cdf_validate_compare.py --baseline <D197.pyd> --candidate <new.pyd> --baseline-dir <frozen-D197-PF-and-realtime-python> --sizes 2048 65536 100000 1000000 --rounds 3 --repeats 3 --output <json>`。
交互確認は `--sizes 1000000 --rounds 5 --repeats 5 --paired`。Release/CUDA12.8/MSVC/compute7.5/C++20/no fast math。

## Exact CPU normalization for uniform particle-pool weights (2026-10-14)

**100万→50万→100万粒子の切替往復を25.59→19.68ms、約23.1%短縮した。**
PR posteriorを含む切替は35.11→32.41ms、約7.7%短縮。
PR処理と毎epochの切替を含む連続5epochは178.57→144.97ms、約18.8%短縮。
直前D196の6source/PE/PF Python/realtimeを凍結し、同じPEでCPU処理だけを比較した。

全有限値1D FP64、N>=65536、NumPy error modeがwarn/ignoreでRuntimeWarning-as-errorがない既存条件の中で、
private copy→元と同じmaxを保ち、先頭・中央・末尾==maxの場合にだけminで全要素が等しいことを確かめる。
均一ならsubtract後の値は±0、expは全て正確な1、sumは配列サイズNなので、同じFP64の1/Nをprivate bufferへ直接fillする。
NumPyの逐次cumsumを元と同じ順序で実行する。CDFをi/Nで近似したりGPU並列累積に置き換えたりしない。
符号付き0/極端有限値/denormにも同じCDFの全bitを確認した。
非均一の場合は元のsubtract→exp→sum→division。先頭・中央・末尾のどれかが最大値と違う通常posteriorでは追加min走査も省く。
小粒子/matrix/非有限値、raise/call/log/print/warning-errorは元のeager path。元配列/view/readonly入力は変更しない。
変更はrealtimeのCPU正規化だけ。native source/PE、全83GPU SASS/resources、state layout、PF runtime/config、
transfer量/stream同期/scratch/left検索/境界clamp/全16状態/均一weight setterを保った。
FP64、粒子数、乱数、測位モデル、許容誤差は同じ。

| Complete synchronized stage, N1000000 | Before ms | After ms | Round-median reduction | Median per-pair reduction | Improving rounds |
|---|---:|---:|---:|---:|---:|
| public_pool_roundtrip | 25.589 | 19.676 | 23.11% | 23.46% | 5/5 |
| public_pool_posterior | 35.106 | 32.408 | 7.69% | 4.43% | 4/5 |
| public_pool_PR_five | 178.571 | 144.965 | 18.82% | 18.89% | 5/5 |
| public_states | 101.807 | 54.999 | 45.98% | 1.19% | 3/5 |
| public_restore | 22.679 | 21.518 | 5.12% | 1.57% | 5/5 |
| PR_control | 5.515 | 5.563 | -0.88% | 0.00% | 1/5 |

同じ論理CPUに置いた既知の両workerで5round×5repeat×6stages=150pairを1callずつ交互に測定。
30round末尾のsource/target全状態/重み/返却snapshotがbit一致。
2048/65536/10万/100万×6stages=24条件の3round×3repeatと、2round×3repeatのprobeも全sample/対照を保存した。
5epochはpredict→12sat PR→ESS→conditional B60 resample→estimate→2回切替→estimate。
初期化/復元/input/IPC/hashは計測外、正規化/CDF/native検査/transfer/GPU処理/同期/配列生成は計測内。非独占host測定。
未変更の全状態getter/host restore/PR単体の変動を本変更へ帰属しない。

CPUだけを取り出した比較も、65,536/10万/100万×uniform/先頭がmaxでない非均一/先頭がmaxの非均一/3点が一致して内部に異なる値を持つ非均一の12条件で各25pair。
同じ自分のprocess/論理CPUで交互実行し、全CDF bitと入力所有権を検査した。GPU転送/検索/gatherは含まない。
負の変動も保持し、以下を切替全体の改善率として扱わない。

| Isolated CPU migration control | Before ms | After ms | Reduction |
|---|---:|---:|---:|
| N65536/uniform | 0.697 | 0.494 | 29.09% |
| N65536/nonuniform_first_not_max | 0.902 | 0.922 | -2.26% |
| N65536/nonuniform_first_max | 0.898 | 0.879 | 2.07% |
| N65536/nonuniform_probe_points_equal | 0.514 | 0.564 | -9.67% |
| N100000/uniform | 0.793 | 0.462 | 41.68% |
| N100000/nonuniform_first_not_max | 1.268 | 1.276 | -0.61% |
| N100000/nonuniform_first_max | 1.079 | 1.134 | -5.13% |
| N100000/nonuniform_probe_points_equal | 1.092 | 1.108 | -1.48% |
| N1000000/uniform | 11.408 | 7.208 | 36.82% |
| N1000000/nonuniform_first_not_max | 14.353 | 16.260 | -13.29% |
| N1000000/nonuniform_first_max | 14.537 | 15.167 | -4.34% |
| N1000000/nonuniform_probe_points_equal | 14.092 | 13.503 | 4.18% |

3,785照合record=3,714出力/71同じ例外。D196の2,727recordを再照合し、736direct CDF/入力所有権と322error policy recordを追加。
1/31/257/65535/65536/65537/10万/100万、23uniform/almost-uniform/nonfinite入力、contiguous/stride/reverse/readonly、
7error policiesとmatrixを照合した。CDF全要素のbit、警告/例外の順序・種類・callback/log/native print textを元式と比較。
旧新とも追加1058test通過。回帰8,237pass。warm/cold/declineを含む実pool90caseへCUDA検査3種を新規実行し0error/leak/hazard。
Python/NumPy traced peakの6caseも全状態snapshot一致、1MiB allocation追跡を確認。C++/GPU確保はtracemalloc対象外。
初期のfirst==max/min分岐は3,693record一致したが、先頭maxの非均一入力へ追加走査の負担が出た。
そのsource/PE/helper/全測定を保存し、先頭・中央・末尾の一致検査で非均一入力を早く除外するよう修正した。
3点が一致しても内部に異なる値がある例を追加し、均一判定は完全なmin走査で確定する。変更後に全検査・測定を再実行。
native PEが同一なのでD196のcuobjdumpログを再利用し、全83body/resourceの一致を監査した。
長時間実データの測位誤差曲線と実機multi-deviceは今回追加評価していない。
均一な重みを持つpool切替に効く改善であり、切替をしない通常PF/raytrace/signal simulationへの速度向上に一般化しない。

[全測定・照合・検査・CPU対照・peak](pf_pool_uniform_gpu_20261014.json)、[コード/patch監査](pf_pool_uniform_device_code_20261013.json)、[完了監査](pf_pool_uniform_completion_check_20261014.json)。

再現: `bench_pf_pool_uniform_compare.py --baseline <D196.pyd> --candidate <same.pyd> --baseline-dir <frozen-D196-PF-and-realtime-python> --sizes 2048 65536 100000 1000000 --rounds 3 --repeats 3 --output <json>`。
交互確認は `--sizes 1000000 --rounds 5 --repeats 5 --paired`。Release/CUDA12.8/MSVC/compute7.5/C++20/no fast math。

## Exact scalar log-weight broadcast in persistent particle-pool migration (2026-10-14)

**100万→50万→100万粒子の切替往復を42.56→35.40ms、約16.8%短縮した。**
PR posteriorの切替も50.59→46.48ms、約8.1%短縮、切替を含む連続5epochは306.19→270.55ms、約11.6%短縮。
3対象とも5/5round改善。100万粒子の均一重みsetter単体は2.92→0.09ms、約97.0%短縮。
直前D195 native/PF Python/realtimeを凍結し、既知の両workerだけを同じ論理CPUに固定した1call旧新交互比較。

CPUで従来どおり `-math.log(N)` を求め、そのFP64 scalar bitsをGPUの各log-weightへコピーする。
新kernelはstoreのみで浮動小数点演算はない。8register、stack/shared/local0、追加GPU allocation/owner/state fieldなし。
元82GPU SASS/resourcesとstate layoutは一致。CPU正規化/CDF累積順序、left検索、上限clamp、粒子選択、乱数/粒子数/測位モデル/許容誤差は保つ。
元setter同様ESSだけを無効化し、既存streamのpending work→fill→同期を順序付ける。位置/速度/共分散とcovariance certificateは保つ。

public `set_uniform_log_weights` はN>=65536で、native Nが一致し、元Python/native array setterを使う場合だけoptional fillを呼ぶ。
元method/native functionのidentityを保持し、instance/class/module/native hookのsetter差替を迂回しない。
small/旧native/欠落hook/未paired/metadata count不一致は元array setter。Poolも元standard uniform methodのidentityを確認し、custom uniform methodは従来setterへ戻る。
cold targetの初期化やold/injected backendの処理は保つ。

| Complete synchronized stage, N1000000 | Before ms | After ms | Round-median reduction | Median per-pair reduction | Improving rounds |
|---|---:|---:|---:|---:|---:|
| public_uniform_weights | 2.920 | 0.087 | 97.00% | 97.02% | 5/5 |
| public_uniform_weights_five | 16.037 | 1.355 | 91.55% | 92.05% | 5/5 |
| public_pool_roundtrip | 42.562 | 35.395 | 16.84% | 15.32% | 5/5 |
| public_pool_posterior | 50.593 | 46.478 | 8.14% | 10.27% | 5/5 |
| public_pool_PR_five | 306.195 | 270.550 | 11.64% | 14.13% | 5/5 |
| public_states | 59.924 | 61.437 | -2.53% | -1.69% | 2/5 |
| public_restore | 23.300 | 23.864 | -2.42% | 0.24% | 1/5 |
| PR_control | 18.120 | 18.216 | -0.53% | -0.94% | 2/5 |

5round×5repeat×8stages=200pair、40round末尾の全状態/weight/返却snapshotがbit一致。
2048/65536/10万/100万×8stages=32条件を3round×3repeatと2round×3repeatでも測定し、全sample/対照/負のroundを保存。
5epochはpredict→12sat PR→ESS→conditional B60 resample→estimate→2回切替→estimateを連続実行。
factory/初期化/復元/input/IPC/hashは計測外。正規化/CDF・setter dispatch・transfer・GPU処理・同期・配列生成は計測内。
全状態getter/host import/PR単体の変動は本変更へ帰属しない。非独占host測定。pool切替をしない通常PF/raytrace/signalへの速度向上には一般化しない。

切替往復でtarget uniform weightのCPU配列と12MB H2Dを省き、2回のscalar kernel引数へ置換する。
source log-weight D2H12MBとCPU CDF H2D12MBは同じで、主要array PCIe転送は36→24MB（kernel launch/parameter overheadを除く）。
Python/NumPy traced peakはPR posterior切替で20.03→16.21MiB。
2048/10万/100万×uniform/posteriorの6caseもsnapshot一致。1MiB NumPy allocationの追跡を確認。C++/GPU allocationはtracemalloc対象外。

2727record=2672出力/55同じ例外。D195の2544を再照合し、156scalar bits/全状態/cache、3queued履歴、24public dispatch（count mismatch3例外）を追加。
1/31/32/33/255/256/257/65535/65536/65537/10万/100万で符号付き0/denorm/極端有限値/Inf/qNaN/sNaN payload/−log(N)を検査。
source-state保全、ESSの再計算、待機中predict/gather、次predict/resample、method/native setter差替を照合。
baseline183pass+4新API skip、candidate187pass、回帰7179pass、1215pool/scalar/CDF caseのmemcheck/racecheck/synccheck0error/leak/hazard。
初期setter測定fixtureのwrapper未初期化は旧版側で発生し保存。public stage prefixを修正後、製品コード/期待値/許容誤差/既存skipを変更せず全照合/測定を再実行。
長時間実データの測位誤差曲線と実機multi-deviceは今回追加評価していない。

[全測定・照合・検査・peak](pf_uniform_weights_gpu_20261014.json)、[コード/patch監査](pf_uniform_weights_device_code_20261013.json)、[完了監査](pf_uniform_weights_completion_check_20261014.json)。

再現: `bench_pf_uniform_weights_compare.py --baseline <D195.pyd> --candidate <new.pyd> --baseline-dir <frozen-D195-PF-and-realtime-python> --sizes 2048 65536 100000 1000000 --rounds 3 --repeats 3 --output <json>`。
交互確認は `--sizes 1000000 --rounds 5 --repeats 5 --paired`。Release/CUDA12.8/MSVC/compute7.5/C++20/no fast math。

## Reuse private CPU normalization work with identical particle-pool CDF (2026-10-14)

**100万→50万→100万粒子の切替往復を48.83→32.16ms、約34.1%短縮した。**
PR posteriorの切替も54.98→39.50ms、約28.2%短縮。
PR処理と各epochの切替を含む連続5epochは294.02→227.87ms、約22.5%短縮。3対象とも5/5round改善。
直前D194のnative/PF Python/realtimeを凍結し、同じnative PEでCPU処理だけを比較した。既知の両workerを同じ論理CPUに置いた1call交互測定。

全重みが有限値、1D FP64、要素数>=65536、NumPy error modeがwarn/ignoreのときだけ適用する。
getterの配列をprivate copyへ写し、元と同じmax→subtract→exp→sum→division→逐次cumsumを行う。
全有限値なら使われないfloor/min/finite抽出/whereを省き、subtract/exp/CDFの出力を同じprivate bufferへ書く。
NumPyのsumの順序、exp、FP64、逐次累積順序、CDF検索/境界clamp/均一weightは維持する。
元配列・readonly/strided/reversed viewは変更しない。
小粒子・matrix・非有限値、raise/call/log/print、RuntimeWarningをerrorにする設定は元のeager式。
CDFをGPU並列累積へ変更することや、近似exp/FP32/粒子数/乱数/測位モデル/許容誤差の変更はない。
native source/PE、全82GPU SASS/resources、state layout、PF runtime/configは完全一致。変更はrealtimeのCPU正規化だけ。

| Complete synchronized stage, N1000000 | Before ms | After ms | Round-median reduction | Median per-pair reduction | Improving rounds |
|---|---:|---:|---:|---:|---:|
| public_pool_roundtrip | 48.825 | 32.161 | 34.13% | 28.73% | 5/5 |
| public_pool_posterior | 54.979 | 39.499 | 28.16% | 25.81% | 5/5 |
| public_pool_PR_five | 294.022 | 227.870 | 22.50% | 23.26% | 5/5 |
| public_states | 43.315 | 47.331 | -9.27% | 0.74% | 2/5 |
| public_restore | 20.180 | 20.277 | -0.48% | -0.83% | 1/5 |
| PR_control | 5.273 | 5.187 | 1.64% | 0.02% | 3/5 |

5round×5repeat×6stages=150pair、30round末尾のsource/target全状態/weight/返却snapshotがbit一致。
2048/65536/10万/100万×6stages=24条件の3round×3repeatと、2round×3repeatのprobeも全sample/対照を保存した。
5epochはpredict→12sat PR→ESS→conditional B60 resample→estimate→2回切替→estimateを連続実行。
factory/初期化/復元/input/IPC/hashは計測外。正規化/CDF・native検査・transfer・GPU処理・同期・配列生成は計測内。
全状態getter/host import/PR単体は変更箇所を通らず、その変動は本改善へ帰属しない。非独占host測定。
pool切替をしない通常PF、raytrace、signal simulationへの速度向上へ一般化しない。

Python/NumPy traced peakはPR posterior切替で35.29→20.03MiB。
2048/10万/100万×uniform/posteriorの6caseもsnapshot一致。1MiB NumPy allocationの追跡を確認した。C++/GPU allocationはtracemalloc対象外。
GPU owner/確保/転送量/既存stream同期/borrowed scratchの扱いは同じ。

2544照合record=2492出力/52同じ例外。D194の2176recordを再照合し、256direct CDF/入力所有権と112error policy recordを追加。
1/31/257/65535/65536/65537/10万/100万、通常値/符号付き0/denorm/極端有限値/混在NaN/Inf/−Inf/raw64bit、readonly/stride/reverseを検査。
CDF全要素のbit、警告/例外の順序と種類、callback/log/native print textを元式と比較。閾値検索後の粒子だけの比較には留めない。
旧新とも追加368test通過。回帰6992pass。warm/cold/declineを含む実pool90caseにmemcheck/racecheck/synccheckを新規実行し0error/leak/hazard。
初期の計測待機はnative stderr captureとworker出力を修正し、元source/PE/helper/logを保存後、検査/全測定を再実行。CDF/例外の期待値・許容誤差・既存skipは緩めない。
長時間実データの測位誤差曲線と実機multi-deviceは今回追加評価していない。

[全測定・照合・検査・peak](pf_pool_normalize_gpu_20261014.json)、[コード/patch監査](pf_pool_normalize_device_code_20261013.json)、[完了監査](pf_pool_normalize_completion_check_20261014.json)。

再現: `bench_pf_pool_normalize_compare.py --baseline <D194.pyd> --candidate <same.pyd> --baseline-dir <frozen-D194-PF-and-realtime-python> --sizes 2048 65536 100000 1000000 --rounds 3 --repeats 3 --output <json>`。
交互確認は `--sizes 1000000 --rounds 5 --repeats 5 --paired`。Release/CUDA12.8/MSVC/compute7.5/C++20/no fast math。

## Exact GPU left search on the retained CPU particle-pool CDF (2026-10-14)

**100万→50万→100万粒子の切替往復を120.85→55.24ms、約54.3%短縮した。**
PR posteriorの切替も110.08→59.74ms、約45.7%短縮。
PR処理と各epochの切替を含む連続5epochは562.36→304.40ms、約45.9%短縮。
直前D193のnative/PF Python/realtime moduleを凍結し、既知の両workerだけを同じ論理CPUへ置いた1call交互比較。3対象とも5/5round改善。

CPU上のfinite処理・weight正規化・exp/sum・逐次NumPy cumsumは元の順序/算術を保つ。
元の閾値生成 `(arange(N)+0.5)/N` をFP64の明示的な丸め加算→除算で再現し、同じCDFを左側境界の二分探索で検索する。
選択indexをsource N−1でclampし、元の全16状態gatherとweight resetを実行する。targetの−log(N) weightも従来どおり。
累積和のGPU並列化・近似検索・FP32・乱数/粒子数/測位モデル/許容誤差の変更はない。
既存81GPU SASS/resource recordが一致し、新検索kernelだけ追加（25register、stack/shared/local0）。

warmな同一backend/deviceでmax(source,target)>=65536かつCDF長さがsource native Nと一致する場合だけ適用する。
小粒子・cold target・旧native・injected/mismatched backend・declineは従来経路。
非有限/非単調CDFはbindingが状態変更前にfalseを返し、元のCPU searchsortedへ戻る。shape/handle不正は変更前に拒否。
同一state/cross-deviceも変更前にfalse。実機multi-device評価は今回行っていない。

| Complete synchronized stage, N1000000 | Before ms | After ms | Round-median reduction | Median per-pair reduction | Improving rounds |
|---|---:|---:|---:|---:|---:|
| public_pool_roundtrip | 120.853 | 55.243 | 54.29% | 54.79% | 5/5 |
| public_pool_posterior | 110.076 | 59.736 | 45.73% | 48.30% | 5/5 |
| public_pool_PR_five | 562.363 | 304.402 | 45.87% | 46.33% | 5/5 |
| public_states | 101.862 | 100.927 | 0.92% | -0.55% | 3/5 |
| public_restore | 23.045 | 23.289 | -1.06% | -0.04% | 2/5 |
| PR_control | 4.033 | 4.064 | -0.76% | -0.31% | 2/5 |

5round×5repeat×6stages=150pair、30round末尾のsource/target全状態/weight/返却snapshotがbit一致。
2048/65536/10万/100万×6stagesの24条件も3round×3repeatで測定し、全sample/負の対照を保存した。
5epochはpredict→12sat PR→ESS→conditional B60 resample→estimate→2回切替→estimateを連続実行する。
factory/初期化/復元/input/IPC/hashは計測外。CPU正規化/CDF生成・native検査・transfer・GPU処理・同期・配列生成は計測内。
全状態getter/host import/PR単体は変更箇所を通らず、その変動を本改善へ帰属しない。非独占host測定。
pool切替をしない通常PF、raytrace、signal simulationへの速度向上には一般化しない。

sourceの既存d_cdf（8×source N）とtargetの既存index arrayを利用し、新規GPU確保/owner/state fieldはない。
source producerを同期後、targetの既存work→CDF H2D→検索→gather→reset→同期を同じstreamに投入する。
source状態/weight/ESS cacheは保つ。借用したsource CDF scratchは戻りまで使用中で、次systematic resampleは元どおり上書きする。
native callerは両stateを戻りまで保持し、他threadから同時変更しない。
100万↔50万の検索input H2Dはancestor6MBからCDF12MBへ増える。weight往復24MBは同じで、主要PCIe転送は30→36MB。
転送量増加とCDF検査も含む全切替で採用判定した。
Python/NumPy traced peakはPR posteriorの切替で39.10→35.29MiB。

2048/10万/100万×uniform/posteriorの6peak caseもsnapshot一致。1MiB NumPy allocationの追跡を確認し、C++/GPU allocationはtracemalloc対象外。

2176照合record=2140出力/36同じ例外。D193の1567recordに576CDF境界×全16状態、9queued/scratch履歴、16サイズ境界/strict NumPy/optional fallback、8weight getter長さを追加。
同値/nextafter前後・重複CDF/ゼロ重み・末尾CDF<閾値・1↔33/31↔33/255↔257/8193→4097/65537→32769を検査。
通常値/符号付き0/denorm/極端有限値/qNaN/sNaN/Inf/raw64bit、input再利用、source保持、target reset/cacheを照合。
source systematic scratch再利用と待機中target pointer swap、KF/predict/次resampleの履歴もhost oracleと一致。
最初のtarget検査はbaseline2203pass+17new API skip、candidate2220pass。追加guardも旧新各8pass、最終版の回帰6624pass。
変更経路と従来poolの1020caseにmemcheck/racecheck/synccheckを順次実行し、error/leak/hazard0。
その後のCDF長さguardはhost側の条件追加だけでnative source/PEは同じ。既存1020caseのinputは追加条件を満たす。
getterを差し替えた長さ不一致は旧版側で成功・当初候補で拒否することを再現/保存し、従来CPU選択→resident ancestor gatherへ戻る条件を加えた。
最終版の追加8caseにもCUDA検査3種を実行して0error/leak/hazard。最終sourceで全照合/24条件/150pair/peak/回帰を再測定し、期待値・許容誤差・既存skipは緩めない。
初期probeも全round/対照と当時のhelper/PEをprivateに保持。長時間実データの測位誤差曲線は今回新規評価していない。

[全測定・照合・検査・peak](pf_pool_cdf_gpu_20261014.json)、[コード/patch監査](pf_pool_cdf_device_code_20261013.json)、[完了監査](pf_pool_cdf_completion_check_20261014.json)。

再現: `bench_pf_pool_cdf_compare.py --baseline <D193.pyd> --candidate <new.pyd> --baseline-dir <frozen-D193-PF-and-realtime-python> --sizes 2048 65536 100000 1000000 --rounds 3 --repeats 3 --output <json>`。
交互確認は `--sizes 1000000 --rounds 5 --repeats 5 --paired`。Release/CUDA12.8/MSVC/compute7.5/C++20/no fast math。

## Resident particle-pool migration with identical CPU selection (2026-10-14)

**100万→50万→100万粒子のpersistent pool切替往復を378.67→144.99ms、約61.7%短縮した。**
PR posteriorからの切替も363.48→124.59ms、約65.7%短縮。
PR処理と各epochの切替を含む連続5epochは1601.29→643.76ms、約59.8%短縮。
3対象とも5/5round改善。直前D192のnative/PF Python/realtime moduleを凍結し、両workerを同じ論理CPUに置いた1call交互比較。

CPU上のweight正規化、有限値処理、exp/sum、CDF、searchsorted、上限clamp、targetの−log(N) weightは元の順序・算術を保つ。
warmな同一backend/device間では、選ばれたint32 ancestorだけをtargetの既存index arrayへ送り、元Megopolis gather kernelでsourceの全16状態を直接コピーする。
新しいGPU kernel・owner buffer・state fieldはなく、全81GPU SASS/resourcesが一致。FP32・近似CDF・粒子数変更・乱数変更・許容誤差変更はない。
source producerを同期した後、targetの既存work→index H2D→gather→weight reset→同期を同じstreamで実行する。
source状態/weight/ESS cacheを保ち、targetのESS/covariance certificateは元importと同様に無効化する。
未初期化target・旧native・injected/mismatched backend・nativeがdeclineした場合は元host経路。cold targetのmean/initを近似しない。
in-place/cross-deviceはnativeが変更前にfalseを返す。実機multi-device検証は今回行っていない。

| Complete synchronized stage, N1000000 | Before ms | After ms | Round-median reduction | Median per-pair reduction | Improving rounds |
|---|---:|---:|---:|---:|---:|
| public_pool_roundtrip | 378.671 | 144.991 | 61.71% | 62.98% | 5/5 |
| public_pool_posterior | 363.482 | 124.589 | 65.72% | 63.76% | 5/5 |
| public_pool_PR_five | 1601.293 | 643.762 | 59.80% | 59.92% | 5/5 |
| public_states | 102.489 | 60.580 | 40.89% | 1.96% | 4/5 |
| public_restore | 23.138 | 22.498 | 2.77% | 3.43% | 4/5 |
| PR_control | 4.038 | 4.040 | -0.03% | -0.03% | 2/5 |

5round×5repeat×6conditions=150pair、30round末尾のsource/target全16state/weight/返却snapshotがbit一致。
切替は実装済み `PersistentParticleRuntimePool.activate` を2回呼び、uniform/PRで重みが偏ったposteriorを検査する。
5epochはpredict→12sat PR→ESS→conditional B60 resample→estimate→2回切替→estimateを連続実行し、全履歴を保持。
factory/初期化/復元/input/IPC/hashは計測外。CPU選択・GPU処理・sync・transfer・配列生成は計測内。
2048/10万/100万粒子×6stagesの18条件を3round×3repeatでも測定し、全sample・対照を保存した。
同じhost上の非独占測定。全状態getter/host import/PR単体は変更箇所を通らず、その速度差は本改善へ帰属しない。
particle pool切替をしない通常PF、raytrace、signal simulationの速度向上へ一般化しない。

切替往復の状態D2H192MB＋H2D192MBを省き、index H2D6MBを追加する。従来も必要だったweight往復24MBは同じで、主要PCIe転送は408→30MB。
Python/NumPy traced peakはPR posterior切替で222.21→39.10MiB。
2048/10万/100万粒子×uniform/posteriorの6peak caseもsnapshot一致。C++ vector/GPU allocationはtracemalloc対象外。
1MiB NumPy allocationの追跡を確認。追加GPU確保はなく、元の全状態getterのtemporary/CPU選択arrayをwarm経路から省ける。

1567照合record=1531出力/36同じ例外。D192の1180recordに、288all16 gather/weight/cache/所有権、9queued source/target履歴、72warm CDF、18cold/refuse fallbackを追加。
1→33/33→1/31↔33/255↔257/8193→4097/65537→32769等で、複製/逆順/random ancestor、通常値/符号付き0/denorm/極端有限値/qNaN/sNaN/Inf/raw64bitを照合。
warmケースではsource full-state getter/target host setterを禁止し、実際にhost往復を回避していることを検査。
queued source/target、pointer swap、KF/predict/次resampleは独立host oracleと一致。oracleのhost readをactual producersより先に行い、producer完了を先に同期してしまわない。
追加15native APIテストはshape/range/NULL/破棄handleの拒否、in-place decline、index形式/使用完了を検査。
baseline1587pass+15new API skip、candidate1602pass、回帰5998pass、memcheck/racecheck/synccheck各1602pass、error/leak/hazard0。
初回fixtureのtarget Python step不一致は旧版側で再現してprivateに保存し、target/oracleの予測回数を揃えた後、全照合/測定を再実行した。期待値・許容誤差・既存skipは緩めない。
長時間実データの測位誤差曲線は今回新規評価していない。

[全測定・照合・検査・peak](pf_pool_gather_gpu_20261014.json)、[コード/patch監査](pf_pool_gather_device_code_20261013.json)、[完了監査](pf_pool_gather_completion_check_20261014.json)。

再現: `bench_pf_pool_gather_compare.py --baseline <D192.pyd> --candidate <new.pyd> --baseline-dir <frozen-D192-PF-and-realtime-python> --sizes 2048 100000 1000000 --rounds 3 --repeats 3 --output <json>`。
交互確認は `--sizes 1000000 --rounds 5 --repeats 5 --paired`。Release/CUDA12.8/MSVC/compute7.5/C++20/no fast math。
private wrapperは既知child PIDだけを同じ論理CPUへ固定する。native C++ callerはsourceを戻りまで保持し、他threadで同時変更しない。

## Large full-state imports using inactive covariance chunks (2026-10-14)

**100万粒子の全16状態復元を28.22→22.84ms、約19.1%短縮した。**
実際のpersistent particle poolで100万→50万→100万粒子を切り替える往復も331.38→313.44ms、約5.4%短縮。
復元単体は5/5、切替全体は4/5round改善。直前D191のnative/Pythonを凍結し、両workerを同じ論理CPUに置いた1call旧新交互測定。

`set_particle_states()` のN>=65536で、既存inactive9N covarianceを16double×floor(9N/16)行のinput stagingとして使う。
H2D→元import kernelを同じstate streamへ順番に投入し、次chunkが前chunkの消費に先行しないようにする。
100万粒子なら562500+437500行の2chunkで、元の追加128MB CUDA allocation/freeが不要。
全chunk後に元reset-weight kernelを1回だけ実行し、stream同期して戻る。
元のESS/covariance certificate無効化、全16state列、重みreset0、入出力ownershipを保つ。
全状態getter、小粒子import、native layout/binding/runtime、全81GPU SASS/resourcesは変更していない。
転送量・数値演算・測位モデル・粒子数・乱数・許容誤差の変更はない。

| Complete synchronized stage, N1000000 | Before ms | After ms | Round-median reduction | Median per-pair reduction | Improving rounds |
|---|---:|---:|---:|---:|---:|
| public_pool_roundtrip | 331.380 | 313.442 | 5.41% | 5.45% | 4/5 |
| public_restore | 28.215 | 22.835 | 19.07% | 21.92% | 5/5 |
| public_state_roundtrip | 69.060 | 60.609 | 12.24% | 10.29% | 4/5 |
| public_states | 44.336 | 43.391 | 2.13% | 0.44% | 3/5 |
| PR_control | 5.987 | 6.623 | -10.63% | -0.25% | 0/5 |

5round×5repeat×5conditions=125pair、25round末尾の全16state/weight/返却snapshotがbit一致。
切替は実装済み `PersistentParticleRuntimePool.activate` を2回呼び、residentな2runtimeのCDF/indices/選択/全状態transfer/weight設定を含む。
source/target両runtimeの全状態とweightをsnapshot比較。factory/初期化/復元/input/IPC/hashは計測外、GPU/CPU/sync/transfer/配列生成は計測内。
2048/65536/10万/100万粒子×7stagesの28条件を3round×3repeatで追加し、gate境界と全状態取得付きPF1/5epochの対照も保存。
全状態getter分割と小粒子import分割の初回trialは1156recordが一致したが、取得/PF処理/小粒子で遅延したため採用しなかった。
大粒子import限定probeでも切替は2round悪化した。これを残し、fresh workerの最初に切替を置いた125交互pairで確認した。
非独占hostのばらつきがあり、切替は1round逆転した。PR対照のround中央値比−10.63%に対し、per-pair差中央値は−0.25%。
round中央値を別々に集計した比とpaired差は同じ統計量ではないので双方を記録し、差の原因は断定しない。
通常の毎epoch PF、全状態readout、raytrace/simulationの速度向上へ一般化しない。

1180照合record=1144出力/36同じ例外。D191の933recordに、128全16bit/ESS/reset/ownership、116checkpointなし独立PFとの履歴、3input形式を追加。
counts65535/65536/65537、100000/100001で境界/tailを検査。通常値/符号付き0/denorm/極端有限値/qNaN/sNaN/Inf/raw64bitも一致。
queued prediction、common/individual covariance、coalesced B4/B60、legacy/systematic、Doppler KF、geometry scratchとの併用を含む。
baseline/candidate1200tests、回帰5596tests、memcheck/racecheck/synccheck各1200testsが通過し、error/leak/hazard0。
長時間実データの測位誤差曲線は今回新規評価していない。

[全測定・照合・検査・不採用trial](pf_state_chunks_gpu_20261014.json)、[コード/patch監査](pf_state_chunks_device_code_20261013.json)、[完了監査](pf_state_chunks_completion_check_20261014.json)。

再現: `bench_pf_state_chunks_compare.py --baseline <D191.pyd> --candidate <new.pyd> --baseline-dir <frozen-D191-python> --sizes 2048 65536 100000 1000000 --rounds 3 --repeats 3 --output <json>`。
交互確認は `--sizes 1000000 --stages public_pool_roundtrip public_restore public_state_roundtrip public_states PR_control --rounds 5 --repeats 5 --paired`。
Release/CUDA12.8/MSVC/compute7.5/C++20/no fast math。同じ論理CPUへの固定はprivate wrapperが既知のchild PIDだけに適用する。

## Particle export using existing inactive covariance scratch (2026-10-14)

**粒子を毎epoch取得する100万粒子のPF連続5epochを105.96→93.62ms、約11.6%短縮した。**
直前D190を凍結したnative/Pythonと1call旧新交互で比較した。

`get_particles()` の4N FP64出力を、既存のinactive 9N covarianceへpackする。
元のpack kernel・引数・stream同期・blocking D2H copyを保ち、毎回の `cudaMalloc/cudaFree` を取り除いた。
100万粒子では追加32MBの一時GPU allocationが不要になる。転送量・CPU返却配列の所有権は同じ。
native state layout・binding/runtimeは変更せず、全81GPU SASS/resourcesが一致。数値演算・モデル・乱数・粒子数・許容誤差の変更はない。

| Complete synchronized public stage | Before ms | After ms | Reduction | Improving rounds |
|---|---:|---:|---:|---:|
| public_particles | 17.312 | 18.595 | -7.41% | 4/5 |
| public_PR_readout_step | 26.599 | 23.016 | 13.47% | 4/5 |
| public_PR_readout_five | 105.959 | 93.620 | 11.64% | 5/5 |
| public_mixture_K8 | 360.459 | 351.912 | 2.37% | 4/5 |
| public_mixture_K16 | 480.379 | 478.563 | 0.38% | 2/5 |
| public_mixture_step | 189.388 | 200.568 | -5.90% | 1/5 |
| PR_control | 6.001 | 6.130 | -2.16% | 2/5 |

100万粒子、5round×8repeat×7conditions=280旧新pairs、35round末尾の全16state/重み/返却snapshotをbit照合。
粒子取得を含むPFはpredict→12sat PR→ESS→conditional B60 resample→estimate→全粒子取得を実行する。
5epochでは連続する全返却粒子とestimate/ESS/resampling履歴を保持・照合する。
初期化/復元/入力作成/IPC/hashは計測外、GPU/CPU処理・同期・D2H・返却配列生成は計測内。
2048/10万/100万粒子×7stagesの21条件を3round×5repeatでも測定し、全sampleを保存した。
K8/K16混合尤度はこのgetterを使う。K4 GPU混合尤度とPR単体は変更箇所を通らず、時間差は対照測定として保持する。
同一host上の非独占測定でばらつきがあり、単体取得・1epochの逆転roundも残している。
通常のPF全用途、raytrace、signal simulationの速度向上へ一般化しない。

### Control rechecks and identical-binary comparison

初回の変更箇所を通らないK4混合PF対照に約5.9%の遅延が出たため、追加150pairで旧新/同一PEを比較した。
旧新K4対照の遅延は約12.6%で再現した。これを無視せず、両workerを同じ論理CPUへ固定し、各stageにfresh workerを使った125pairを追加した。
この条件の粒子取得付き5epochは101.661→90.484ms、10.99%短縮・5/5round改善。ペアごとの短縮率の中央値は9.65%。
K4対照のペアごとの短縮率中央値は−0.38%、PR対照は−0.35%。同一candidate PEのK4対照もround中央値比−9.58%・ペア差中央値−1.42%となった。
同じPEにも差があり、CPU実行位置・非独占hostの変動が対照差に影響したと推測する。原因を一つに断定せず、すべての逆転round/sampleを残す。
round中央値を別々に集計した比は、paired差の中央値と一致しない。両方を記録する。
追加275pair/55snapshotも全出力・状態がbit一致。対象5epochの短縮は通常CPUと同じCPUの両条件で確認した。

| Confirmation stage | Before ms | After ms | Reduction of round medians | Median per-pair reduction | Improving rounds |
|---|---:|---:|---:|---:|---:|
| 追加・通常CPU/old_new/public_particles | 9.016 | 7.251 | 19.57% | 20.74% | 5/5 |
| 追加・通常CPU/old_new/public_PR_readout_five | 110.389 | 107.559 | 2.56% | 11.47% | 4/5 |
| 追加・通常CPU/old_new/public_mixture_step | 189.327 | 213.232 | -12.63% | -4.15% | 0/5 |
| 追加・通常CPU/old_new/PR_control | 6.847 | 6.859 | -0.18% | -0.16% | 2/5 |
| 追加・通常CPU/same_new/public_mixture_step | 173.959 | 174.363 | -0.23% | 1.05% | 3/5 |
| 追加・通常CPU/same_new/PR_control | 5.529 | 5.528 | 0.00% | 0.05% | 4/5 |
| 追加・同じ論理CPU/old_new/public_PR_readout_five | 101.661 | 90.484 | 10.99% | 9.65% | 5/5 |
| 追加・同じ論理CPU/old_new/public_mixture_step | 164.995 | 170.686 | -3.45% | -0.38% | 1/5 |
| 追加・同じ論理CPU/old_new/PR_control | 4.067 | 4.082 | -0.36% | -0.35% | 2/5 |
| 追加・同じ論理CPU/old_new/public_particles | 14.394 | 13.409 | 6.85% | 20.39% | 5/5 |
| 追加・同じ論理CPU/same_new/public_mixture_step | 209.068 | 229.089 | -9.58% | -1.42% | 1/5 |


933照合record=897出力・36同じ例外。D190の779recordを実際の凍結旧版/新版で再実行した。
追加80件は通常値/符号付き0/denorm/極端有限値/qNaN/sNaN/Inf/raw64bitのexportと、全16state/weight/ESS不変・返却配列独立性。
追加74履歴はcommon/individual covariance、queued predict、coalesced B4/B60、legacy、systematic、Doppler KF、geometry tableとのscratch再利用。
10万粒子のindividual B60 packed gatherを含み、getterを呼ばない独立PFともその後の全状態をbit照合した。
baseline/candidate953tests、回帰5343tests、memcheck953tests/0errors/0bytesleaked、racecheck953tests/0hazards、synccheck953tests/0errorsが通過。
長時間実データの位置誤差曲線は今回新規評価していない。
比較fixtureでsystematic API名を取り違えた初回はprivate trialへ保存し、修正後に全照合・測定を再実行した。

[測定・全sample・照合・検査](pf_particle_readout_gpu_20261014.json)、[GPUコード・再構成patch](pf_particle_readout_device_code_20261013.json)、[完了監査](pf_particle_readout_completion_check_20261014.json)。

再現: `bench_pf_particle_readout_compare.py --baseline <D190.pyd> --candidate <new.pyd> --baseline-dir <frozen-D190-python> --sizes 2048 100000 1000000 --rounds 3 --repeats 5 --output <json>`。
1call確認は `--sizes 1000000 --rounds 5 --repeats 8 --paired`。Release/CUDA12.8/MSVC/compute7.5/C++20/no fast math。

## Exact FP64 mixture geometry on GPU with unchanged NumPy exp/log (2026-10-14)

**100万粒子・4候補の混合位置尤度を含むPF1epochを242.43→175.12ms、約27.8%短縮した。**
2候補の連続5epochも1093.29→734.85ms、約32.8%短縮。直前D189との増分を1call旧新交互で測定した。


`position_mixture_update` の2-4候補・大粒子条件で、座標差分/二乗/x-y-z加算/0.5乗算/σ²除算/log_mix減算をGPUへ移した。
全nodeを `__dsub_rn/__dmul_rn/__dadd_rn/__ddiv_rn` でFP64の丸めごとに実行し、FMAへのcontractを防ぐ。
小さな混合weightのlogと、粒子ごとのexp/logsumexp・成分方向sum・最終weight加算は元NumPyのまま。
候補/粒子数・モデル・乱数・許容誤差は変更しない。FP32/近似exp/log/距離の逆数近似は使わない。

4候補以下のN×K tableは既存のinactive 9N covarianceへ置ける。resamplerはこれを必要時に全上書きしてから使う。
非有限座標flagは既存reduction/pinned result bufferを使い、log-weight・active covariance・host ESS cacheを変更しない。
block thread0が16 FP64係数をsharedへ準備し、tailを含む全threadがbarrier後にcoalesced tableを書き込む。
新3kernelは各regs40/shared128/stack0/local0、元78GPU SASS/resourcesとnative state layoutは一致。
追加CUDA allocationやGPU owner bufferはなく、元の全粒子getterのCUDA一時allocate/freeを省ける。
100万粒子の通常updateは元の全粒子/重み48MB往復から、2候補32MB、3候補40MB、4候補48MB（各+flag4bytes）へ。
係数は128byteのinline kernel input。flag/tableのD2Hは完了まで同じstate streamを同期する。

### Complete synchronized public calls: one-call alternating confirmation

| Stage | Before ms | After ms | Reduction | Improving rounds |
|---|---:|---:|---:|---:|
| `N1000000/public_mixture_K2` | 192.707 | 117.589 | 38.98% | 5/5 |
| `N1000000/public_mixture_K3` | 194.125 | 120.494 | 37.93% | 5/5 |
| `N1000000/public_mixture_K4` | 228.318 | 156.390 | 31.50% | 5/5 |
| `N1000000/public_mixture_step` | 242.427 | 175.117 | 27.77% | 5/5 |
| `N1000000/public_mixture_five` | 1093.288 | 734.852 | 32.79% | 5/5 |
| `N1000000/public_mixture_K8` | 276.909 | 265.508 | 4.12% | 3/5 |
| `N1000000/public_position_control` | 0.207 | 0.204 | 1.40% | 5/5 |
| `N1000000/PR_control` | 5.562 | 5.403 | 2.85% | 5/5 |

100万粒子、5round×5repeat×8conditions=200旧新pairs。40round末尾の全16state/重み/返却snapshotもbit一致。
毎timed log-weight/returnを照合。復元/initialize/入力生成/IPC/hashは時間外で、pairedは各round/processに1warmup。
stepはpredict/12衛星PR/4候補mixture/ESS/条件付きB60 coalesced resample/estimate。
fiveは同じ処理を2候補で連続5epoch実行して全履歴を返す。GPU/CPU転送と全stream同期を含む。
8候補、単Gaussian position、PRはGPU geometryを使わない対照。負の値と改善round数も保持し、非専有hostの変動を利得とみなさない。
別の27conditions（N2048/100000/1000000×9stages、3×3）、15memory cases、予備14conditions（2×3）も全件保存。
効果の対象は2-4候補の混合位置尤度を使うPF。通常PR-only PF・raytrace・実データ全pipelineの同割合の改善は主張しない。

### Incremental traced host allocation peak

| Mixture call | Before MiB | After MiB |
|---|---:|---:|
| `N1000000/public_mixture_K2` | 54.46 | 38.68 |
| `N1000000/public_mixture_K3` | 54.85 | 46.44 |
| `N1000000/public_mixture_K4` | 53.47 | 53.43 |
| `N1000000/public_mixture_K8` | 53.52 | 53.52 |
| `N1000000/public_mixture_K16` | 53.63 | 53.63 |

tracemallocで追跡したcall中の追加host peak。復元/入力/persistent PF stateと検査用state dumpは計測外。
総process RSSやGPUメモリ削減率ではない。2/3候補はhostへ全4-stateの代わりに2/3 FP64成分を返す。

### Exactness and fallback

旧新779recordsが一致（output/変換743、matching exception36）。D189のpublic536casesを実旧新で再照合し、
追加216GPU tableは元の3D NumPy数式とuint64比較。旧版に新APIはないため旧workerが独立3D参照を作り、候補が実GPU tableを照合する。
N1/31/32/33/255/256/257/8193/32769、2/3/4候補、signed zero・denormal・極端finite座標/σ²・geometry overflow/
NaN payload/sNaN/Inf、非対象state/weight NaN payloadとcached ESSの保持を確認。
追加27historiesでpredict/common・individual B4/B60 gather/Doppler KF/legacy resample後のscratch再利用を、
同じGPUで新geometryを挟まない独立PFと比較。返却table・inputの直後の変更もdevice stateに影響しない。

非有限particle座標はflag確認後にNoneを返し、元CPU経路でNaN payload・value behaviorを保持する。
small input、1候補・5候補以上、σ² nonfinite、strict NumPy raise/call/log/printとRuntimeWarning-as-error、旧native/injected backendはD189 CPU処理を使う。
新APIはshape/finite/log_mix/positive finite σ²を検証。native source/C++ binding/既存runtimeへ追加した部分以外は元byteを保持する。
旧版536tests PASS・新API263skip、候補799tests PASS。既存PF/BVH/signal等5,189 regression PASS。
799testsのCUDA memcheck/racecheck/synccheck errors/hazards/leaks0、GPU検査は順次実行。
実データ長時間trajectoryの位置誤差曲線は新規評価していない。

再現: `dist/run_pf_mixture_geometry_final_20261014.ps1`、`dist/run_pf_mixture_geometry_paired_20261014.ps1`、
`dist/run_pf_mixture_geometry_checks_20261014.ps1 -Mode baseline_unit|candidate_unit|regression|memcheck|racecheck|synccheck`。
[全measurement/validation/checks](pf_mixture_geometry_gpu_20261014.json)、[SASS/resources/再構築patch](pf_mixture_geometry_device_code_20261013.json)、
[完了audit](pf_mixture_geometry_completion_check_20261014.json)、[比較helper](bench_pf_mixture_geometry_compare.py)、
[table/再利用cases](pf_mixture_geometry_cases.py)、[263native tests](../tests/test_pf_mixture_geometry.py)。
source/PE/helper/raw/checksは `dist/pf_mixture_geometry_adopted`、直前D189は `dist/pf_mixture_geometry_baseline` に保存。

## Ordered 2D NumPy mixture likelihood with bounded particle chunks (2026-10-14)

**100万粒子の混合位置尤度を含むPF1epochを386.77→241.33ms、約37.6%短縮した。**
2候補を使う連続5epochも1649.31→1274.28ms、約22.7%短縮。どちらも1call旧新交互の全5roundで改善した。


`position_mixture_update` の粒子×候補×3差分と二乗配列、およびlogsumexpの大きな中間配列を減らした。
65,536粒子×成分を超える通常設定では粒子を分割し、2つの2次元FP64作業配列でx/y/zの差分・二乗・加算を順番に実行する。
元と同じ0.5乗算/σ²除算/混合log weight/最大値/exp/sum/log/最終加算を、独立NumPy ufuncのまま処理する。
成分方向のsum/logsumexp順序を保持し、候補数・粒子数・尤度・乱数・精度を変えない。全成分を評価し、近似や枝刈りはない。
chunkの行数は `max(1,65536//n_components)`。小さい入力は元の3D処理を保持し、全粒子の状態/重み転送は同じ。
変更はPythonのこのmethodだけ。他のPython AST/ソースbyteとnative source/配置済み・build済みPEは一致する。
同じPEのため全78GPU SASS/resourcesも同じで、D188のcuobjdumpをPE SHA照合後に再利用した。追加GPU allocation/kernel/転送はない。

### Complete public calls: one-call alternating confirmation

| Stage | Before ms | After ms | Reduction | Improving rounds |
|---|---:|---:|---:|---:|
| `N1000000/public_mixture_K2` | 233.021 | 171.266 | 26.50% | 5/5 |
| `N1000000/public_mixture_K4` | 343.457 | 213.733 | 37.77% | 5/5 |
| `N1000000/public_mixture_K8` | 573.037 | 284.504 | 50.35% | 5/5 |
| `N1000000/public_mixture_K16` | 1047.934 | 391.761 | 62.62% | 5/5 |
| `N1000000/public_mixture_step` | 386.768 | 241.332 | 37.60% | 5/5 |
| `N1000000/public_mixture_five` | 1649.307 | 1274.285 | 22.74% | 5/5 |
| `N1000000/public_position_control` | 1.442 | 1.437 | 0.37% | 4/5 |
| `N1000000/PR_control` | 5.658 | 5.671 | -0.24% | 0/5 |

100万粒子、5round×5repeat×8conditions=200旧新pairs。40round末尾の全16state/重み/返却snapshotもbit一致。
各timed log-weight/returnをhash照合し、復元/initialize/入力生成/IPC/hashは時間外。
pairedは各roundの各processに1warmup後、旧新1callずつ交互に同期込みで測定。5epochはpredict/12衛星PR/2候補mixture/
ESS/条件付きB60 coalesced resample/estimate/全履歴を含む。stepは同じ処理の4候補・1epoch。
単Gaussian positionとPRは混合尤度を使わない対照。負の値と改善round数も保持し、非専有hostの変動を効果に数えない。
別の24conditions（N2048/100000/1000000×8stages、3×3）・12memory casesも全件保存。
対象はこのCPU混合位置尤度を使うPF。通常PR-only PF・raytrace・実データ全pipelineへの同割合の改善は主張しない。

### Incremental traced host allocation peak

| Mixture call | Before MiB | After MiB |
|---|---:|---:|
| `N1000000/public_mixture_K2` | 161.24 | 54.46 |
| `N1000000/public_mixture_K4` | 252.79 | 53.47 |
| `N1000000/public_mixture_K8` | 465.40 | 53.52 |
| `N1000000/public_mixture_K16` | 892.64 | 53.63 |

tracemallocの追加host peak。入力生成/復元/persistent PF stateと検査用の全状態dumpは計測外。
NumPy allocationの追跡を1MiB配列で検証した。総process RSSやpersistent GPU memoryの削減率ではない。
GPU↔CPUの全状態/重み往復、元native getterのCUDA一時allocate/freeは保持する。

### Exactness, exceptional policies and verification

新536recordsが実旧版Python/候補Python・同一native PEで一致。
CPU456は1/2/3/4/7/8/13/33/65成分、chunk境界・端数、signed zero/全IEEE bit pattern/NaN payload/
極小・極大/±Inf/NaN sigma/無効mixture weightを、保持した元3D NumPy oracleとuint64比較。
GPU40はN33/257/8193/32769/65537×2/4/8/13成分×通常/非有限で、全16state・重み・ESS・estimate・spreadを照合した。
入力拒否32と数学error policy8も照合。536内訳はoutput/変換500、matching exception36。

初期trialは値が一致したが、strict NumPyで元のsubtract overflowよりlog divide-zeroが先に出るケースがあった。
そのsource/helper/raw24条件を保存し、`raise/call/log/print` とRuntimeWarningをerrorにするfilterでは元のeager処理へ戻す。
8追加casesはerror型/message・callback/log順・重み不変/出力を比較。最終source/helperでvalidationと全測定をやり直した。
NumPy warningの通常の位置や数はchunk処理に伴い異なり得る。数学errorを例外/handlerにする設定は元経路で保持する。

追加536testsは旧版・候補双方でPASS。既存PF/BVH/signal等を含む4,926 regression PASS。
536testsのCUDA memcheck/racecheck/synccheckもerrors/hazards/leaks0。全GPU検査を順次実行した。
実データ長時間trajectoryの位置誤差は新規評価していない。

再現: `dist/run_pf_mixture_coordinates_final_20261014.ps1`、`dist/run_pf_mixture_coordinates_paired_20261014.ps1`、
`dist/run_pf_mixture_coordinates_checks_20261014.ps1 -Mode baseline_unit|unit|regression|memcheck|racecheck|synccheck`。
[全measurement/validation/checks](pf_mixture_coordinates_gpu_20261014.json)、
[device proof/再構築patch](pf_mixture_coordinates_device_code_20261013.json)、
[完了audit](pf_mixture_coordinates_completion_check_20261014.json)、
[比較helper](bench_pf_mixture_coordinates_compare.py)、[元3D oracle/cases](pf_mixture_coordinates_cases.py)、
[536tests](../tests/test_pf_mixture_coordinates.py)。
native PE/全source/helper/raw/checksは `dist/pf_mixture_coordinates_adopted`、直前D188は `dist/pf_mixture_coordinates_baseline`、
strict error修正前は `dist/pf_mixture_coordinates_error_policy_trial` に保存。

## Reuse exact ESS while particle weights are unchanged (2026-10-14)

IMU/RBPF等で `get_ess()` の直後に `resample_if_needed()` が同じESSを再計算していた。
重みが変わらない間は、最初のGPU計算で返ったFP64 scalarをstateごとの独立host slotへ保持して再利用する。
ヒット時も元のstream同期を残し、pending predict/shift/covariance処理の完了・エラー境界を維持する。
PR/DD/GMM/搬送波/Doppler/KF/position likelihood、initialize/reset/import/set weights、3種類のresampleの
全16native入口で無効化する。estimate/spreadによるGPU partial・pinned resultの上書きはcacheに影響しない。
CPU側にscalarとvalid flagを追加し、既存fieldと関数宣言を維持。native structへfieldを追加するためC++ clientは対応headerで再buildする。
native device weight pointerへ直接書くclientはcacheを無効化する必要がある。Python公開APIの変更はない。
追加GPU allocate/transfer/kernelはなく、元の78GPU functionsのSASS/resourcesおよびGPU sourceは完全一致。

### Complete synchronized calls: one-call alternating confirmation

| Stage | Before ms | After ms | Reduction | Improving rounds |
|---|---:|---:|---:|---:|
| `N1000000/public_ESS_pair` | 1.0426 | 0.5637 | 45.93% | 5/5 |
| `N1000000/public_PR_one_sat_step` | 9.9212 | 9.2560 | 6.70% | 5/5 |
| `N1000000/public_PR_step` | 11.2570 | 10.6538 | 5.36% | 5/5 |
| `N1000000/public_RBPF_five` | 50.1749 | 47.8354 | 4.66% | 5/5 |
| `N1000000/public_PR_KF_step` | 38.5761 | 38.1207 | 1.18% | 5/5 |
| `N1000000/public_covariance_PR_five` | 54.4564 | 52.2030 | 4.14% | 5/5 |
| `N1000000/public_ESS_single_control` | 1.6407 | 1.6054 | 2.15% | 3/5 |
| `N1000000/public_PR_single_ESS_control` | 11.8563 | 11.9757 | -1.01% | 2/5 |

100万粒子、5round×20repeat×8conditions=800旧新pairs。40round末尾の全16状態/重み/返却snapshotも一致。
各pairの旧新順を交互にし、毎timed weight/returnをhash照合。入力生成/initialize/復元/IPC/hash/3warmupは時間外。
単独ESSとPR single-ESSは再利用をしない対照。負の値・改善round数も保持し、非専有hostの変動を一般的な利得と解釈しない。
別のstage-batch測定24conditions（N2048/100000/1000000×8stages、5×20）と予備16conditions（3×5）を全件保存。
効果の対象は重み更新間にESSを重複照会する公開PF処理。KF等が占める時間の大きさで全体の割合は変わる。

### Exactness and verification

旧新25,227records（出力/変換24739、入力拒否488）がbit一致。元D187の24,987baseline recordsは
旧source/PE/変更していないvalidator helperのSHA照合後に再利用し、候補は全recordsを実測。
追加240は実旧新binaryで実行。8粒子数・全16mutation・6weight-preserving操作と8weight分布を検証した。
各操作の後に元のGPU ESS再計算を強制し、repeat、estimate/spreadによる共有buffer再利用、
signed zero/極端値/全NaN/±Inf/混在重みでもuint64で一致。query前後の全状態/重みが不変。
systematicは1回の実resample後に強制再計算して比較し、既存の独立systematic draw変動をESSのoracleに使わない。
粒子数・観測数・FP64式・reduction tree・乱数・resample・閾値・許容誤差は変更しない。

追加240testsは実旧版・候補双方でPASS。既存PF/BVH/signalを含む4,390 regression PASS。
追加240testsのCUDA memcheck/racecheck/synccheckはerrors/hazards/leaks0、GPU検査はすべて順次実行。
実データ長時間trajectoryの位置誤差曲線は新規評価しておらず、同じ割合の実データ全体利得は主張しない。

再現: `dist/run_pf_ess_cache_final_20261014.ps1`、`dist/run_pf_ess_cache_paired_20261014.ps1`、
`dist/run_pf_ess_cache_checks_20261014.ps1 -Mode baseline_unit|unit|regression|memcheck|racecheck|synccheck`。
[全measurement/validation/checks](pf_ess_cache_gpu_20261014.json)、
[device codeと再構築patch](pf_ess_cache_device_code_20261013.json)、
[完了audit](pf_ess_cache_completion_check_20261014.json)、
[比較helper](bench_pf_ess_cache_compare.py)、[交互helper](bench_pf_ess_cache_paired_compare.py)、
[独立replay cases](pf_ess_cache_cases.py)、[240 tests](../tests/test_pf_ess_cache.py)。
source/PE/helper/raw/checksは `dist/pf_ess_cache_adopted`、直前D187は `dist/pf_ess_cache_baseline` に保存。

## Uniform velocity covariance broadcasts without full particle round trips (2026-10-14)

**毎epoch共分散を設定する100万粒子PFの5epoch公開処理を、1call旧新交互で595.51→54.70ms、約90.8%短縮した。**
この測定はsynthetic covariance-guided PFで、設定/predict/PR/ESS/条件付きB60 resample/estimate/全履歴返却を含む。
共分散を設定しない通常PF、都市raytrace全体、実データIMU runへの同じ割合の向上は主張しない。

従来の `set_velocity_covariance` は全N×16状態とN重みをD2Hし、CPUで9共分散columnを書き、
全状態をH2Dしてweight reset後に元重みをH2Dで戻していた。100万粒子では状態/重み往復272MBとCUDA一時allocate/freeを伴う。
新しいnative `pf_device_set_velocity_covariance(state,covariance)` は72bytesの9FP64係数をkernel parameterへcopyし、
blockのthread0がshared係数を準備、全threadがbarrierを通った後にflat9Nのrow-major covarianceだけをcoalesced writeする。
新規CUDA allocate/H2D/D2H state/weight transferはない。1kernel追加、regs26/stack/local0/shared80bytes（9double+alignment）。
7個の他state成分、全log-weight、粒子/観測数、乱数、API既存入力のreshape/finite条件を保持。
元の最終weight restoreと同じstate stream同期を終えてから返り、inputを直後に変更しても係数が変わらない。

uniform covarianceが9成分すべて一致することをnative setterが保証し、既存 `covariance_uniform` certificateをtrueにする。
その後のpredictはD184の元FP64 Cholesky式を共有する経路、必要時resampleは既存の共通共分散経路を使える。
同じ9値を配るだけで共分散の近似/対角化/対称化は追加しない。任意のfinite非対称/負の成分も元APIと同じ扱い。
importやDoppler KFは元のcertificate invalidationを保持。古いnative/injected backendには元roundtrip fallbackが残る。
PFのstruct layoutは変更せず、元77GPU functionsのSASSとresourcesは完全一致。追加はbroadcast1kernel。

### Complete synchronized calls: sequential one-call paired confirmation

| Stage | Before ms | After ms | Reduction | Improving rounds |
|---|---:|---:|---:|---:|
| `N1000000/public_covariance_set` | 100.7809 | 0.3496 | 99.65% | 5/5 |
| `N1000000/public_covariance_predict` | 106.4418 | 2.7669 | 97.40% | 5/5 |
| `N1000000/public_covariance_PR_step` | 138.3741 | 10.9244 | 92.11% | 5/5 |
| `N1000000/public_covariance_PR_five` | 595.5055 | 54.6984 | 90.81% | 5/5 |
| `N1000000/public_covariance_KF_step` | 163.6148 | 41.9690 | 74.35% | 5/5 |
| `N1000000/PR_control` | 6.8375 | 6.8434 | -0.09% | 1/5 |

5round×20repeat×6conditions=600旧新pair。3warmups/初期化・復元・IPC・input生成・hashは計測外。
毎timed log-weight/returnを比較し、round末尾16-state/weights/returnの30pair snapshotも一致。
共分散setter自体は100.7809→0.3496ms。
測定の主眼は、このsetterを含む公開step/連続5epoch全体。Doppler KF併用stepも163.6148→41.9690ms。
別のstage-batch測定21conditions（N2048/100000/1000000×7stages、5×20）と予備12conditions（3×5）をJSONへ全件保存。
小粒子条件・共分散未設定のPR対照・負の値も保持。非専有hostの対照差から一般的な無退行は証明しない。
異なるround方式の大きな中央値差を選んで最終効果を報告しない。

### Exactness and verification

旧新24,987records（出力/変換24499、入力拒否488）が一致。
既存23,963baseline recordsはsource/PE/変更していないvalidator helperを照合して再利用し、候補は全recordsを実測。
追加1,024recordsは旧版・候補の両binaryで実行。保持した元Python roundtripを強制する独立oracleに、
9covariance/他7state/weightsをuint64で照合し、続く元individual predictorとの出力も比較。
N1/31/32/33/255/256/257/8193、部分block、8matrices、C/Fortran/strided、signed zero/NaN payloadの非対象state/weights、
極小/極大finite covariance、4predict parameters、queued KF/サイズ増減/input mutation/resample、入力拒否を確認。
許容誤差を使わず、浮動小数点の式やRNG drawを変更しない。
初期fixtureのzero sigma_posは**旧版の公開入力検証で計測前に拒否**され、fixtureのみtiny-positive1e-300へ修正してarchiveした。

元public method/injected-backendの257新testsは旧版・候補双方で合格。
新native API専用13testsは旧版moduleにentrypointがなくskip、候補では全270tests合格（shape/finite/null/destroyed/input conversionも含む）。
PF/raytrace/signal/IMU adapterの4,150回帰tests合格。
270testsのmemcheck/racecheck/synccheckすべて0errors/hazards、leak0bytes。
BVH D169とsignal D185のsource/build/installed PEを保持。長期実データの測位誤差分布は今回新規測定していない。

再現: `dist/run_pf_covariance_broadcast_final_20261014.ps1`、`dist/run_pf_covariance_broadcast_paired_20261014.ps1`、
`dist/run_pf_covariance_broadcast_checks_20261014.ps1 -Mode baseline_unit|unit|regression|memcheck|racecheck|synccheck`。
CUDA12.8 / MSVC Release / compute7.5 / C++20 / no fast math。
source/PE/actual baseline Python/helper hashes、全cases/rounds/samples/pairs、private archive manifest:
[`pf_covariance_broadcast_gpu_20261014.json`](pf_covariance_broadcast_gpu_20261014.json)。
SASS/resources/5source filesのbyte復元patch:
[`pf_covariance_broadcast_device_code_20261013.json`](pf_covariance_broadcast_device_code_20261013.json)、
[`pf_covariance_broadcast_20261013.patch`](probes/pf_covariance_broadcast_20261013.patch)。
測定helper: [`bench_pf_covariance_broadcast_compare.py`](bench_pf_covariance_broadcast_compare.py)、
[`bench_pf_covariance_broadcast_paired_compare.py`](bench_pf_covariance_broadcast_paired_compare.py)、
[`pf_covariance_broadcast_cases.py`](pf_covariance_broadcast_cases.py)。
完了audit: [`pf_covariance_broadcast_completion_check_20261014.json`](pf_covariance_broadcast_completion_check_20261014.json)。

## Doppler likelihood-only specialization rejected after complete-step measurements (2026-10-14)

**速度更新gain<=0に限って元Doppler kernelの尤度prefixを専用化したが、PF全体の有効な短縮がなく不採用。元のsource/build/installed PEを復元した。**
正gain、NaN gain、65超観測、KF、他weight/query/predict/resampleは元producerを保持して比較。
候補は6関数を追加し、元77GPU関数のSASS/resourcesは完全一致。尤度prefixの式・演算順をbyte復元で確認。
レジスタ数は70→56（33..64観測は76→58）、速度WLS normal matrix分のstack160bytesを除けたが、
粒子ごとの幾何・clock drift・尤度の実行計算は同じままで、資源削減を速度向上と取り違えない。
観測/粒子数・FP64式・Philox・resampling・API/stream/allocation/転送を削減/変更しない。

候補と旧版の実binaryを別の永続workerで順次実行、3round×5repeat、roundごとに旧新順を交互に変更。
stage入力とstate ownerは再利用し、初期化/復元・3warmups・IPC・hashは計測外。単体updateは終了syncを含み、
公開stepはpredict/PR/Doppler/ESS/条件付きB60 resample/estimate/返却を含む。5epochは連続状態と全履歴返却を含む。
全timed weights/returnsをhashし、batch末尾の16-state/weights/returnも比較。
非専有hostの予備判定であり、1callごとの交互確認・新規候補の回帰/sanitizersへは進めていない。

| Stage | Before ms | Candidate ms | Reduction | Improving rounds |
|---|---:|---:|---:|---:|
| `N100000/native_Doppler12_gain0_update` | 2.4288 | 2.4526 | -0.98% | 1/3 |
| `N100000/public_Doppler12_gain0_step` | 5.7689 | 5.4356 | 5.78% | 2/3 |
| `N100000/public_Doppler12_gain0_five` | 11.4061 | 11.6886 | -2.48% | 1/3 |
| `N100000/native_Doppler32_gain0_update` | 2.7043 | 2.7385 | -1.26% | 1/3 |
| `N100000/public_Doppler32_gain0_step` | 3.9564 | 3.9548 | 0.04% | 3/3 |
| `N100000/public_Doppler12_gain0.25_step` | 3.0068 | 3.0827 | -2.52% | 1/3 |
| `N100000/PR_control` | 0.5517 | 0.5347 | 3.08% | 3/3 |
| `N1000000/native_Doppler12_gain0_update` | 12.1365 | 14.2936 | -17.77% | 1/3 |
| `N1000000/public_Doppler12_gain0_step` | 19.8587 | 19.7064 | 0.77% | 3/3 |
| `N1000000/public_Doppler12_gain0_five` | 97.8722 | 97.9501 | -0.08% | 1/3 |
| `N1000000/native_Doppler32_gain0_update` | 25.8550 | 25.8348 | 0.08% | 3/3 |
| `N1000000/public_Doppler32_gain0_step` | 35.6913 | 35.5896 | 0.28% | 3/3 |
| `N1000000/public_Doppler12_gain0.25_step` | 25.8433 | 25.8629 | -0.08% | 1/3 |
| `N1000000/PR_control` | 4.4870 | 4.4856 | 0.03% | 3/3 |

100万粒子の公開12観測step19.8587→19.7064ms（約0.8%）、5epoch97.8722→97.9501ms（約0.1%悪化）。
32観測公開stepも35.6913→35.5896ms（約0.3%）。小粒子での一部正値やregister差だけでは追加6kernelを維持しない。
単体12観測の悪化や正gain/PR不変対照も、全14条件と全round/sampleを上表/JSONに残す。

旧新の1,925records（state/weight/queued growth/input mutation/resample）がbit一致。
N1/127/128/129/8193、0/1/3/4/8/9/16/17/24/25/32/33/48/49/64/65観測、
gain +0/-0/negative/-Inf/NaN/tiny-positive/.25/+Inf、zero/negative/NaN weights、非finite geometryを確認。
非正gainは元CUDA fmax clampでも0になり、NaN gainはhost guardを通らず元producerを選ぶ。
元GPU clamp `fmax(0,NaN)` でWLS tailが止まることも**旧版・候補の両方で**検証し、保持した元producerとのuint64比較を追加。
許容誤差は使わず、測位精度低下を許して採用しない。長期実データの測位誤差分布は新規評価していない。

復元sourceを通常rebuildし、元77SASS/resources一致を確認してから、以前採用したexact PEをbuild/installedへ復元。
installedは候補を導入していない同じ旧PEであり、PF共通予測とDoppler既存781testsが合格。
前回のPF D184、raytrace D169、signal D185のsource/build/installedを照合して保持。

再現: [`bench_pf_doppler_likelihood_compare.py`](bench_pf_doppler_likelihood_compare.py)、
`dist/run_pf_doppler_likelihood_probe_20261014.ps1`（対応baselineへ記録したpatchを適用してsourceを候補へ戻し、rebuild後にoriginal frozen binaryと比較。現在の製品sourceは不採用に戻している）。
CUDA12.8 / MSVC Release / compute7.5 / C++20 / no fast math。
候補source/pyd/helper/全validation/全timing/private archive manifest:
[`pf_doppler_likelihood_rejected_20261014.json`](pf_doppler_likelihood_rejected_20261014.json)。
SASS/resourcesとbyte復元patch:
[`pf_doppler_likelihood_device_code_20261013.json`](pf_doppler_likelihood_device_code_20261013.json)、
[`pf_doppler_likelihood_20261013.patch`](probes/pf_doppler_likelihood_20261013.patch)。
復元確認: [`pf_doppler_likelihood_rejected_completion_check_20261014.json`](pf_doppler_likelihood_rejected_completion_check_20261014.json)。

## Static NAV above256 channels uses the original arithmetic without descriptor branches (2026-10-14)

**257チャネル・各260,000サンプル・4epochの公開処理を、1call旧新交互で単一生成群約16.0%、バッチ生成約16.7%短縮した。**
ここで単一生成群は4回または32回の `generate_epoch` と `np.stack`、バッチは1回の `generate_epochs`。
チャネル変換/検証/H2D/kernel/sync/D2H/返却を含み、入力生成・初期化・warmup・IPC・hashは除外。
非専有hostのため、kernel資源や非交互の結果だけで採用判断せず、旧新1callずつ交互の5round×20repeatを使う。

前回の33..256単一epoch/<=256 batch専用化を保持し、valid channel数（batchは最大）が256を超え、
packed NAV配列が空のときだけ既存static templateの未cache版を新しくinstantiate。
変更はhostの2dispatchだけ。既存11GPU関数のSASSとresourcesは完全に同じ。新しい単一関数は33register、batchは43register、
stack/local/sharedはいずれも0。元の未cache動的対応版は44/54register。この資源差からoccupancyや速度を推定しない。
全channel入力順のFP64時刻/chip/phase、FP32 multiply/addのrounding、Philox seed/sequence/draw、
サンプル数/チャネル数/return ownership/API/stream/allocation/転送を保持する。
動的NAV、空、<=256経路は元producer。frequencyをCPUで再計算したりchannelを落としたりしない。

### Measured complete groups: sequential per-sample paired confirmation

| Stage | Before ms | After ms | Reduction | Improving rounds |
|---|---:|---:|---:|---:|
| `public_single_plain/epochs4/samples26000/channels257` | 29.8850 | 24.2936 | 18.71% | 5/5 |
| `public_single_plain/epochs4/samples26000/channels512` | 44.0139 | 39.0329 | 11.32% | 5/5 |
| `public_plain/epochs4/samples26000/channels257` | 26.5213 | 22.0889 | 16.71% | 4/5 |
| `public_plain/epochs4/samples26000/channels512` | 44.1812 | 42.1340 | 4.63% | 5/5 |
| `public_single_plain/epochs32/samples2600/channels257` | 184.0981 | 177.4880 | 3.59% | 4/5 |
| `public_plain/epochs32/samples2600/channels257` | 142.5422 | 133.7514 | 6.17% | 5/5 |
| `public_single_plain/epochs4/samples260000/channels257` | 54.1514 | 45.5011 | 15.97% | 5/5 |
| `public_plain/epochs4/samples260000/channels257` | 50.9313 | 42.4090 | 16.73% | 5/5 |
| `public_single/epochs4/samples26000/channels257` | 31.5215 | 32.1724 | -2.06% | 3/5 |
| `public_plain/epochs4/samples26000/channels65` | 6.1361 | 6.6475 | -8.34% | 0/5 |

1,000旧新pairの入力と返却IQのhashを全件確認。短い32epochや不変のdynamic/65channel対照を含め、負の値も表に残す。
不変65channel対照の初回約8%悪化（0/5改善）を受け、同じ旧版同士2条件×100pairと旧新対照再検査2条件×100pairを追加。
同じ旧版の65channelは10.5995→10.1892ms（2/5改善）と揺れ、
旧新再検査は4.2691→4.1067ms（2/5改善）。
この校正を初回の悪化を消すために使わず、全round/pairをJSONに保存した。非専有host上の対照差から一般的な無退行は証明しない。
非交互14条件の全round・全sampleと予備8条件もJSONに保存した。非交互native batch257/26,000サンプルは
12.4908→13.2583msと悪化しており、公開callの交互結果へ置き換えて隠さない。
広いチャネル数全般、都市trajectory、測位処理全体への同率の速度向上は主張しない。

### Exactness and verification

旧新1,215records（出力/変換1135、同じ入力拒否80）が完全一致。
257/258/512/1024channels、1/255/256/257/4097samples、noise on/off、最大uint64 seed、
符号/0を含むstatic bit、混合GNSS、極端なIF/phase、empty/invalid/dynamic切替、部分blockと返却ownershipを確認。
新経路同士の比較だけでなく、各channelへ同じ値の1bitをpositive1Hzで渡して**保持した元dynamic producer**を選び、
旧新双方でIQのuint32を比較した。初期oracle fixtureの0Hzは旧版の入力検証で拒否され、計測前に1Hzへ修正。
失敗fixture/helper/logはprivate archiveに保持し、製品ソースはこの修正で変えていない。
新規92testsを旧版・新版とも実行し合格、PF/raytrace/signalの3,864回帰tests合格。
92testsのmemcheck/racecheck/synccheckはいずれも0errors/hazards、leak0bytes。
PF共通共分散予測D184とBVH D169のsource/build/installed binaryを保持。
許容誤差を緩めず、精度検証はテストしたIQ/metadataのbit同一まで。長期実データの測位誤差分布は今回追加測定していない。

再現: `dist/run_signal_wide_static_final_20261014.ps1`、`dist/run_signal_wide_static_paired_20261014.ps1`、
`dist/run_signal_wide_static_checks_20261014.ps1 -Mode baseline_unit|unit|regression|memcheck|racecheck|synccheck`。
CUDA12.8 / MSVC Release / compute7.5 / C++20 / no fast math。
全validation/round/pair/source/pyd/helper hashとprivate archive manifestは
[`signal_wide_static_gpu_20261014.json`](signal_wide_static_gpu_20261014.json)。
SASS/resourcesとbyte復元patchは
[`signal_wide_static_device_code_20261013.json`](signal_wide_static_device_code_20261013.json)、
[`signal_wide_static_20261013.patch`](probes/signal_wide_static_20261013.patch)。
測定helper: [`bench_signal_wide_static_compare.py`](bench_signal_wide_static_compare.py)、
[`bench_signal_wide_static_paired_compare.py`](bench_signal_wide_static_paired_compare.py)。
完了audit: [`signal_wide_static_completion_check_20261014.json`](signal_wide_static_completion_check_20261014.json)。

## Common covariance prediction shares exact GPU Cholesky factors (2026-10-14)

**100万粒子の共通共分散RBPF/PR・5epoch公開処理を、1call旧新交互の確認で約10.7%短縮した。**
この5epochでは共通共分散が継続し、予測/PR/ESS/必要時B60 resample/estimateと全履歴返却を含む。
初期化直後だけを毎回測ってsteady KFへ一般化しない。Doppler KFやimportで共分散が個別になった後は元producer。
velocity KF予測かつ既存 `covariance_uniform` 保証がある場合だけ新producerを選択する。
非KF予測/個別共分散/速度sample/noise/観測/RNG/粒子数/祖先/ESS条件/全query tree/API/streamは元のまま。

元kernelのFP64 Cholesky式をcopyし、blockのthread0が**そのblockの最初の粒子**を読み、同じ順序のsqrt/divide/fmax/FMAを実行。
6coeffをFP64 shared48bytesへ格納し、全threadがbarrierに参加してから元のtail returnへ進む。
毎blockでparticle0を読む案は、block0のin-place Q更新とraceになるため使わない。
Philox init/normal draw順、guideの明示FMA、粒子別noise/位置/clock/対角Q更新を保持。
共通covarianceからの係数を共有するだけで、乱数sampleや粒子を共有/削減しない。
新規allocation/H2D/D2H/launchはなく、新kernel58register/stack/local0/shared48bytes。元KFは60register/shared0。
全76旧GPU SASS body/resourceとkernel sourceを保持してnew1kernelを追加。register数だけをoccupancyの証明にしない。

**23,963旧新一致**（23,531出力等、432既存input棄却）。既存23,499はsource/pyd/全validator helper SHAを照合して元記録を参照し、
追加464は旧新版とも実行。さらにcandidateの全validatorを実行して比較した。
新common casesは、同じ全16状態をimportしてcommon保証を明示的に無効化した**保持原producer**とuint64で独立比較する。
N1/31/32/33/255/256/257/8193、signed zero/tiny/huge/有限inputからのInf、切替、最大seed、8queued changing coeff、resample後、
Doppler KF後を含む。追加**512 tests**は実原版/新版で通過。既存PF/BVH/signalを含む**3,772 regression PASS**。
512testsのmemcheck0errors/0bytes leaked、racecheck0hazards、synccheck0errors。
source全体を元kernel-copy+原factor移動+load6個+host predicateで再構成し、元FP64式とRNG/per-particle更新を確認。
初回racecheckではhazards0でも、uniform systematicのN8193/seed0で旧新oracleの祖先128件が異なり、511pass/1failとなった。
実原版pyd同士でも同じ128祖先・7state成分の差を再現した。製品resamplerは変更していない。
予測比較を1回のsystematic結果16成分のexact cloneから開始し、clone前後/予測後のuint64一致を確認するoracleへ修正。
bit基準を緩めず、修正前fixture/失敗/原版diagnosticをarchiveし、修正後512旧新・3,772回帰・3sanitizersを再実行した。

本計測3N×9stage=**27条件・5×20**、100万粒子の**8条件800pair**、2,048粒子の**4条件400pair**を保存。
別persistent workerでGPU順次、初期化/restore/3warmup/inputs構築/IPC/hashは時間外。
各timed log weight/returned ESS・estimate・resample履歴をhash照合し、batch末・各paired round末で全16状態/重み/返却を照合。
native/public共通・imported予測、単一全step、連続5epoch、測定KF直後初回、未変更PR対照を含む。
全negative/control/round/sampleを残し、非占有hostの条件別時間を混ぜたり過去速度比を乗算しない。

| 100万粒子・1call交互確認 | 原版 (ms) | common factor (ms) | 中央値短縮 | 改善round |
|:---|---:|---:|---:|---:|
| native共通KF予測 | 6.2240 | 4.1444 | 33.41% | 5/5 |
| 公開共通KF予測 | 6.2480 | 4.2171 | 32.51% | 5/5 |
| native共通RBPF/PR全step | 10.8967 | 9.7584 | 10.45% | 5/5 |
| 公開共通RBPF/PR全step | 11.3629 | 10.1673 | 10.52% | 5/5 |
| native共通RBPF/PR・5epoch | 53.4926 | 47.5159 | 11.17% | 5/5 |
| 公開共通RBPF/PR・5epoch | 57.7528 | 51.5635 | 10.72% | 5/5 |
| 未変更imported KF予測 | 3.6484 | 3.6467 | 0.05% | 4/5 |
| 未変更PR尤度 | 5.6540 | 5.6634 | -0.17% | 2/5 |

小粒子数の全体は変動が大きく、一律の利得を主張しない。本計測N2048 native5epochは中央値のみだと負でも4/5round改善だったため、
個別の1call交互確認とpaired差分も保存した。

| N2048・交互確認 | 原版 (ms) | common factor (ms) | 改善round |
|:---|---:|---:|---:|
| N2048/native_PR_RBPF_five | 1.0603 | 1.0284 | 5/5 |
| N2048/public_PR_RBPF_five | 1.5326 | 1.5034 | 3/5 |
| N2048/native_predict_KF_imported | 0.0412 | 0.0444 | 1/5 |
| N2048/public_PR_KF_step | 0.4423 | 0.4299 | 4/5 |

accuracy toleranceを変更していない。長時間実データの測位誤差分布は今回追加評価していない。
[27条件・1,200pair・23,963比較・全archive](pf_common_predict_gpu_20261014.json)、
[元76body/resource・追加1kernel・exact patch](pf_common_predict_device_code_20261013.json)、
[最終source/build/installed・検査・BVH/signal保持](pf_common_predict_completion_check_20261014.json)。
再現helper `bench_pf_common_predict_compare.py` は `--baseline --candidate --baseline-dir --validate --sizes --stages --rounds 5 --repeats 20 --output`、
`bench_pf_common_predict_paired_compare.py` は同reportを `--reference` に渡し、SHA一致のときだけvalidationを参照する。
CUDA12.8 / MSVC Release / compute7.5 / C++20 / no fast math。PF個別gather・BVH parallel mirror chunks・signal static single/batchを保持。

## Static NAV single epochs retain the exact batch producer (2026-10-14)

**33..256 valid channelの静的NAV単一epochを専用producerへ分岐し、65ch・各260,000 samplesの8個別生成全体を約25.2%短縮した。**
前回のstatic NAV batch版からの追加差分であり、過去の速度比を乗算しない。
host変更はglobal descriptor・33..256で `packed_nav_bits.empty()` の場合のkernel選択1箇所だけ。
既存のstatic templateをsingleとして実体化するため、全既存kernel sourceと全10 GPU SASS body/resourceを保持し、new1kernelを追加。
inline1..32、dynamic NAV、全invalid/空、257以上、全batchは元の経路。
過去NAV bufferが残っても現在のpacked列を確認し、invalid channelのNAVは元と同じfilterで除く。

元のFP64時間/chip整数剰余/phase、sincos、ordered FP32積/加算、Philox seed/sequence/noise、sample/channel数を保持。
新規allocation/transfer/launch/APIはない。新global static singleは42→31register、stack/local0、shared2048bytesは同じ。
register数だけをoccupancyや速度の証明にはしない。
**1,045旧新比較一致**（965 IQ/metadata/変換等、80 native例外）、追加202比較は保持batch producerとの独立uint32照合を含む。
新**205tests**は実原版/新版で通過。新205casesと既存PF/BVH/signalを含む**3,260 regression tests PASS**。
新205testsのmemcheck0errors/0bytes leaked、racecheck0hazards、synccheck0errors。
count1/31/32/33/65/255/256/257、partial blocks、noise有無、最大seed、static bit-3/0/2、空NAV、極端位相/cancellation、
dynamic/static/above256切替、invalid NAV/返却owner/closeを確認。precision toleranceを緩めていない。

本計測13条件・5×20、同source/pyd/helperの1group交互確認9条件・5×20=900旧新pair。
singleの1groupは8回または32回の元単一epoch呼出しを順に行い、結果の `np.stack` まで含む。
別persistent processでGPU順次、stage inputs・1reference/3warmup group・IPC/hashは時間外。
validation/変換/packing/H2D/kernel/同期/D2H/ordinary owning返却/集約は時間内。
batch対照は元の一括generate_epochsであり、singleをbatchに置き換えて測っていない。
都市単一epoch経路も測り、LOS/反射/channel metadata/実CUDA IQ/trajectory yieldを含めたが、今回の小channel条件は未変更。
非占有hostで未変更経路も時間変動するため、stage-batchと1group交互結果を保存。全negative/control/round/sampleを残す。

| 1group交互確認 / 各epoch260,000 samples | 原版 (ms) | static single (ms) | 中央値短縮 | 改善round |
|:---|---:|---:|---:|---:|
| 公開・static33ch・8個別epoch | 50.9582 | 41.8284 | 17.92% | 4/5 |
| 公開・static65ch・8個別epoch | 88.3894 | 66.1128 | 25.20% | 5/5 |
| native・static65ch・8個別epoch | 36.4618 | 28.6990 | 21.29% | 5/5 |
| 公開・static256ch・8個別epoch | 75.0863 | 59.9061 | 20.22% | 5/5 |
| 未変更・公開static12ch | 26.5099 | 26.5158 | -0.02% | 3/5 |
| 未変更・公開dynamic65ch | 49.6748 | 46.3444 | 6.70% | 3/5 |
| 未変更・公開static65ch batch | 32.5035 | 32.3608 | 0.44% | 3/5 |

短い32epoch×2,600samplesの交互再確認は、公開29.5006→28.5378ms（4/5改善）、
native16.6236→14.1505ms（5/5改善）で、本計測のnative悪化を再現しなかった。
未変更dynamic対照にも約6.7%の中央値変動があるため、小差を効果と読み替えず、短いepoch、小channel、都市全体に一律の利得は主張しない。長時間実データの測位誤差分布は今回追加評価していない。
最初の1..256無条件single専用化は1,045一致でも12ch全体の安定利得がなく、source/pyd/helper/9条件3×5をfreezeし棄却。
最終版ではinlineを元へ戻した。棄却版に追加full regression/sanitizersは実行しておらず、採用版の結果を流用しない。

[13条件・900交互pair・1,045比較・archive](signal_static_single_gpu_20261014.json)、
[既存10body/resource一致・追加1kernel・正確なpatch](signal_static_single_device_code_20261013.json)、
[最終source/build/installed・検査・PF/BVH保持](signal_static_single_completion_check_20261014.json)、
[unrestricted棄却試行](signal_static_single_unrestricted_rejected_20261014.json)。
再現helper `bench_signal_static_single_compare.py` は `--baseline --candidate --baseline-dir --stages --rounds 5 --repeats 20 --output`、
`bench_signal_static_single_paired_compare.py` は同full reportを `--validated-report` に渡し、source/pyd/helper SHA一致のときだけ1,045記録を参照する。
CUDA12.8 / MSVC Release / compute7.5 / C++20 / no fast math。PF individual gather、BVH parallel mirror chunks、static NAV batchを保持。

## Static NAV epoch batches remove unreachable descriptor work (2026-10-14)

**8epoch×260,000 samplesの静的NAV・65チャネル公開信号生成を、1callずつ旧新版交互の確認で約24.8%短縮した。**
同確認の1,200三角形・12衛星・8epochの都市シミュレーション全体は約17.1%短縮。
sample/channel数、元のFP64時間/code/phase、FP32積/入力順加算、Philox seed/sequence/noise、ordinary owning NumPy IQ/metadataを保持する。

Host packingで全valid channelのNAV列が空と分かり、max valid countが1..256の**batchだけ**専用producerを使う。
新2kernelは元producerのcopyから到達しないdynamic-NAV分岐だけを除いたもの。
static nav_bitを同じ式へ渡す。packed NAVが空ならvalid各channelのnav_bit_count==0が保証される。
NAVあり・全row空・257ch以上・single epochは元の経路。過去NAV bufferが残っていても、現在のpacked列とcountを毎回使う。
coeff準備/barrier/partial-block return、正確なcode整数剰余、phase最終FMA、sincos、noise順は保持する。
新規buffer/確保/転送/launch/API追加なし。GPU registerは両cache幅で51→40、stack/local0、shared256/2048 bytesは元と同じ。
register数だけをoccupancyや実測速度の証明にしない。

**843旧新比較一致**（763 IQ/metadata/変換等、80既存native例外）。元のsingle producerとの独立uint32照合も含む。
static bitの-3/0/2、空NAV array、31/32/33/255/256/257、partial blocks、noise有無、最大seed、極端位相/cancellation、
dynamic→static→fallback切替と返却所有を確認。新**104tests**は実原版/新版の両方で通過。
PF/BVH/signal等**3,055 regression tests PASS**。104casesのmemcheck0 errors/0bytes leaked、racecheck0 hazards、synccheck0 errors。
元の全8 GPU SASS body/resourceを保持し、new kernel2だけを追加した。sourceはkernel-copy差分とhost predicateまで全体を再構成して確認。

本計測**12条件・5×20**、同じsource/pyd/helperの交互確認**6条件・5×20=600旧新pair**。
各timed IQ/metadataをhash/bit照合。別persistent processでGPUは順次、stage input構築・参照1call/3warmup・IPC/hashを時間外、
validation/変換/packing/H2D/kernel/同期/D2H/ordinary owning返却を時間内にする。
都市全体はLOS/反射/channel metadata/実CUDA IQ/trajectory yieldを含み、CPU mesh/BVH構築は時間外。
GPU/hostは非占有。stage-batchでは未変更above256にも差があったため、交互確認と対照を保存し、一律の速度比や過去の速度比の乗算はしない。

| 1call交互確認 / 各epoch260,000 samples | 原版 (ms) | static NAV (ms) | 中央値短縮 | 改善round |
|:---|---:|---:|---:|---:|
| 公開 static12ch・8epoch | 19.5643 | 15.7667 | 19.41% | 5/5 |
| 公開 static65ch・8epoch | 53.2742 | 40.0439 | 24.83% | 5/5 |
| native static65ch・8epoch | 37.7445 | 28.0969 | 25.56% | 5/5 |
| 都市全体・1,200三角形・8epoch | 27.8604 | 23.1097 | 17.05% | 5/5 |
| 未変更 dynamic NAV・公開12ch | 20.1946 | 19.9816 | 1.05% | 5/5 |
| 未変更 above256・native257ch | 68.1908 | 68.1575 | 0.05% | 3/5 |

短い32epoch×2,600 samplesの公開/都市に一律の利得は主張しない。本計測の全負値/round/sampleも保存した。
精度toleranceは緩めていない。長時間実データの測位誤差分布は新規評価していない。

### 同時に見送った転送2案

事前warm GPU activityでは8epoch×260,000 samples・12chのkernel2.8553ms、D2H3.0319msで、D2Hはactivityの約51.5%。
activityを通常wall timeの速度比に読み替えない。Device列を含むraw traceはprivateに保持した。
1..32MiBの繰り返し出力へ再利用pinned stagingを使い、普通のowning NumPy配列へCPU memcpyする案と、
8..32MiBの返却配列を転送中だけregister/unregisterしてCPU copyを省く案を比較した。
各737比較が一致、全8 GPU body/resourceも同一だったが、CPU copy/registration/releaseを含む公開・native・都市全体は遅くなり両方不採用。
各8条件3×5とsource/pyd/helper/patchをarchiveし、製品の元転送/所有権/確保へ戻してstatic NAV案を測った。
これら棄却trialには追加full regression/sanitizersを実行しておらず、static NAV案の検査結果を流用しない。

[全12条件・600交互pair・843比較・転送2案・source/pyd/helper/archive](signal_static_nav_gpu_20261014.json)、
[元8body/resource一致・new2kernel・正確なpatch](signal_static_nav_device_code_20261013.json)、
[最終source/build/installed・検査・PF/BVH保持](signal_static_nav_completion_check_20261014.json)。
再現helperは `bench_signal_static_nav_compare.py` と `bench_signal_static_nav_paired_compare.py`。
前者は `--baseline` / `--candidate` / `--baseline-dir` / `--stages` / `--rounds 5 --repeats 20 --output`。
後者は同full reportを `--validated-report` に渡し、source/pyd/helper SHAが同じ場合だけ843記録を参照する。
CUDA12.8 / MSVC Release / compute7.5 / C++20 / no fast math。PF individual gatherとBVH parallel mirror chunksを保持した。

## BVH construction trials preserve trees but lack effective speedup (2026-10-14)

**CPU BVH構築の3案を実ビルドで比較し、すべて不採用。**
製品側のBVH source/build/installed pydは、直前採用のparallel mirror chunksへbyte/SHA単位で復帰した。
PF individual covariance gatherとsignal epoch batchもsource/build/installedとも保持した。

1. 3軸のbin集計を1回のgeometry走査へ統合。各axis内の三角形順・union operand順・bin式・後続SAH/sort/splitは保持した。
2. `1/bin_width`と積から整数bucketを求め、境界から2^-40以上離れて同じ整数になると保証できる場合だけ元の除算を省く。
   binary64 RNのproduct-vs-division absolute errorを2^-46未満（subnormal reciprocalを含む）と有理数で確認。
   全rounding directionにも2^-44未満のboundを確認した。境界・非有限・範囲外は元の除算/変換/clamp。
   1,261,890生成組のうちguardを通る310,361組の整数結果が元の除算と一致した。
3. 親prefixの左子境界は、その子の入力を元と同じascending順にunionした値なので、scratch上書き前にparent localへcopyして再利用。
   reverse suffixはsigned-zero tieでascending scanとbitが異なり得るため、右子/rootのscanは保持した。新規heap確保なし。

各案で**97旧新ケースの構築木・sorted indices・反射/LOSがbit一致**し、全timed返却もbit一致。
random/degenerate/coplanar/同centroid/signed zero/極小・極大scale/ECEF/入力layout/変更meshと返却所有を含む。
各案の**全10既存GPU SASS body/resourceが同一、新kernel0**。元の候補数・FP64反射/距離・threshold・訪問順/tieを変更しない。
除算案のproductはbin分類の補助であり、境界計算や反射出力を近似する用途には使わない。

1,200/12,000/120,000三角形×7stageの**各21条件・3round×5call**を、旧新版1callずつ順次・交互測定。
native/public build、構築+初回trace+close、構築/再構築+実CUDA都市IQ、未変更warm controlを含む。
CPU BVH構築、public snapshot/並べ替え、必要なGPU upload/kernel/sync/D2H/closeを時間内、mesh生成・出力hashを時間外とした。
以下は同じ案・同じprotocol内の値で、別試行の絶対時間を組み合わせない。

| 公開BVH構築 | 原版 (ms) | 試行 (ms) | 中央値短縮 |
|:---|---:|---:|---:|
| 3軸bin走査の統合 / 12,000三角形 | 22.0999 | 23.2800 | -5.34% |
| 3軸bin走査の統合 / 120,000三角形 | 214.6418 | 233.4643 | -8.77% |
| 同じ整数binを保証する除算短縮 / 12,000三角形 | 16.5107 | 17.0205 | -3.09% |
| 同じ整数binを保証する除算短縮 / 120,000三角形 | 268.1341 | 283.8069 | -5.85% |
| 左子bounds再利用・5×10確認 / 12,000三角形 | 13.8697 | 13.6755 | 1.40% |
| 左子bounds再利用・5×10確認 / 120,000三角形 | 203.5549 | 206.0427 | -1.22% |

3軸統合と除算短縮は、公開構築/再構築全体に安定した利得がなく見送った。
左子boundsの初期3×5は12,000三角形の公開構築19.8309→18.4285msなど改善も見えたが、未変更warm controlにも変動があった。
同source/pydを固定したCPU build6条件・**5×10**の追加確認では、12,000三角形の公開構築は約1.4%短縮に留まり、120,000では悪化。
同確認のown-thread scheduled cyclesも約1.3%短縮/約1.2%悪化で、大きなalgorithmic費用削減を確認できないため追加複雑性を採用しない。

Windowsの`process_time_ns`は短いcallを0と返すことがあり、その中央値の速度比はnullとする。
`QueryThreadCycleTime`はcalling threadだけのscheduled cyclesを読み、wall timeと別に全sample/roundを保存した。
cyclesをmsや測位速度比に換算しない。診断追加前のprobe/helper/pydも別archiveで固定した。
GPU/hostは非占有。棄却後の追加full regression/CUDA sanitizersは行っておらず、以前の検査結果をこれらtrialへ流用しない。
精度toleranceは緩めていない。長時間実データの測位誤差分布は新規評価していない。

[3案の全21+21+21条件・追加6条件・source/pyd/helper SHA・97case各比較・archive/patch](bvh_build_trials_rejected_20261014.json)、
[最終復元・PF/BVH/signal・公開表記の整合](bvh_build_trials_rejected_completion_check_20261014.json)。
再現helperは `bench_bvh_build_bins_compare.py` / `bench_bvh_build_bin_reciprocal_compare.py` / `bench_bvh_build_left_bounds_compare.py`。
共通baselineはparallel mirror chunksのsource5/pydを固定し、`--counts 1200 12000 120000 --rounds 3 --repeats 5`。
左boundsの追加確認は `--cpu-build-only --rounds 5 --repeats 10`。CUDA12.8 / MSVC Release / compute7.5 / C++20 / no fast math。

## PF individual-covariance gather reuses its former source (2026-10-13)

**100万粒子の公開predict→PR update→Doppler KF→ESS/条件付きB60 resample→estimateを、1callずつ旧新版交互の確認で約6.6%短縮した。**
粒子数・観測数・FP64演算・Philox系列・受理順・祖先・共分散9成分・APIは原版のまま。
事前のwarm GPU activityでは100万粒子/KFの祖先選択1.9604ms、gather4.9290ms、reset0.0288msで、gatherが約71%を占めた。
このactivity時間をwall-time利得やoccupancyの証明に読み替えない。

個別共分散では、先に元と同じ9成分mapping/lane順で全共分散をdestinationへcopyする。
その完了後、同じstream上でformer source covariance9*Nのうち8*Nを、7状態列の64byte packed行へ転置する一時領域に再利用する。
前回採用したcopy-only pack/gather kernelを使い、全pointer swap・primary SoA・row-major covariance・positive-zero weight resetを保持する。
新規device buffer・確保・解放・host transferは追加しない。個別共分散の起動数は元の3から4になり、共通共分散の3起動は保持する。

**個別共分散の新経路はN≥100,000かつB≥60に限定する。**
無制限適用の試行では2,048粒子/native PR+KFが0.2750→0.2967ms、
10万粒子/B4が0.2046→0.2158msと遅くなったため、その領域は元のgather/resetを使う。
閾値は測定できた領域へ保守的に限定する判断で、全機種の最適境界を主張しない。
共通共分散のpositive-iteration経路、B0/negative、legacy/systematic resamplerは元の動作を保持する。
無制限試行の全27条件と500交互pairもreportに残した。

**23,499旧新比較一致**（23,067出力/状態/軌跡等、432既存入力棄却）。
原版source/pyd/全validator dependency SHAを確認して保存済み記録を使い、同じ全validatorを最終candidateで実行した。
全75既存GPU SASS body/resourceは一致。追加copy-only kernelは18register・stack/shared/local0で、元の共分散copy loopとmappingが同一。
独立整数Philoxの祖先と全16状態bit/NaN payload/negative zeroを確認する**31新規tests**は、原版/新版で通過。
99,999/100,000/100,001粒子の閾値・端数、最大seed、queued B1/B60/B0/B257/B4/B61、legacy/systematic/common/reimport切替を含む。
前回361testsと既存6 lifecycle casesを合わせた**398 targeted tests**は原版/新版で通過。
PF/BVH/signal等**2,951 regression tests PASS**。398casesのmemcheck0 errors/0bytes leaked、racecheck0 hazards、synccheck0 errors。

最終source/pyd/helperを固定し、2,048/10万/100万粒子×9stageの**27条件・5round×20**を別persistent workerで順次・交互計測。
100万粒子5条件を**1 measured callずつ交互の5×20**で確認した。各timed log-weight/returned outputと全stateのbatch末/paired-round末snapshotはbit一致。
CPU入力・ownerをstageごとに一度だけ構築し、initialize/restore・3warmup・IPC・hashを時間外にする。完了同期を含むwall timeを測る。
交互確認は全stateをround末に照合し、各callではlog-weight/returnを照合する。全stateを各callで照合した前回protocolと絶対時間を混ぜない。

| 1call交互確認 / 100万粒子 | 直前 (ms) | 個別copy+packed (ms) | 中央値短縮 | 改善したround |
|:---|---:|---:|---:|---:|
| 個別共分散 B60 resample | 6.1189 | 3.8861 | 36.49% | 5/5 |
| native PR+KF update step | 42.0025 | 39.2681 | 6.51% | 5/5 |
| 公開 PR+KF update step | 42.6055 | 39.7989 | 6.59% | 4/5 |
| 未変更の単独 PR weight | 4.0393 | 4.0608 | -0.53% | 3/5 |
| 未変更の共通共分散 B60 | 3.2806 | 3.1471 | 4.07% | 3/5 |

最終stage-batchの公開/native PR+KFは双方5/5roundで改善。1call交互確認の独立round中央値はnative5/5、公開4/5で改善した。
公開100pairのうち77pairでcandidateが速く、対応pairの時間差中央値は2.7001ms短縮、各roundでも5/5で正だった。
独立中央値で悪化した公開round（53.4583→56.2544ms）も削除しない。元のround中央値と対応pair差の両方を記録し、後者を前者の「5/5改善」と言い換えない。
この一致した全体/対応pairの改善と、B60の100/100pair短縮を根拠に採用。
10万粒子の公開PR+KF stage-batchは4.2477→4.2761msで、全体への効果は小さい。
未変更controlの変動や小規模の値を一律の利得に含めない。GPU/hostは非占有で、過去の速度比を乗算しない。
測位演算の近似や粒子削減は行わない。長時間実データの測位誤差分布は新規評価していない。

CUDA12.8 / MSVC Release / compute7.5 / C++20 / no fast math。
[全27条件・500交互pair・無制限適用試行・23,499比較・source/pyd/helper SHA・verification/archive](pf_individual_gather_gpu_20261013.json)、
[原版75body/resource一致・copy-only1kernel・正確なpatch](pf_individual_gather_device_code_20261013.json)、
[最終source/build/installed/検査整合](pf_individual_gather_completion_check_20261013.json)。
再現helperは `bench_pf_individual_gather_compare.py` と `bench_pf_individual_gather_paired_compare.py`。
full helperは `--baseline` / `--candidate` / `--baseline-dir` / `--validate` / `--sizes 2048 100000 1000000` / `--rounds 5` / `--repeats 20` / `--output`。
paired helperは同じfull reportを `--reference` に渡す。原版記録と最終版のsource/pyd/helper SHA一致を必須とする。
BVH mirror chunksとsignal epoch batchの採用source/build/installedも保持した。

## PF common-covariance resampling uses packed state rows (2026-10-13)

**100万粒子の公開predict→PR update→ESS/条件付きB60 resample→estimateを、1callずつ旧新版交互の確認で約21.6%短縮した。**
元のPhilox系列・全粒子/観測数・FP64尤度/予測・受理順・祖先・9共分散成分・APIを保つ。
GPU activityの事前確認では100万粒子/common-cov B60で祖先選択1.9795ms、gather3.5515ms、reset0.0282ms。
この1callのGPU kernel時間の約64%が、散在する7 SoA状態列のgatherだった。GPU activityをwall-time利得やoccupancyの証明に読み替えない。

共分散が全粒子で同じと既存flagで保証される間、未使用のdestination covariance9*N doublesのうち8*Nを一時領域に使う。
position3/velocity3/clockの7 doubleを、64byteに揃えた行へcopy-only kernelで転置し、選択された行のdouble2を読み元のSoA出力へcopyする。
primary SoA・active covarianceの9成分は保持し、新規device buffer/確保/解放/host transferを追加しない。
paddingは0で初期化する。共有tileは16,448bytes/block、全tail laneがbarrierへ参加し、valid8*Nだけを書き込む。
祖先選択が同じstreamで完了した後なので、gather内で元のpositive-zero resetを行える。kernel起動数は元と同じ3。
state import/Doppler KFで既存のuniform保証が無効になる場合とB0/negativeは、元のgather/resetへ戻る。
共分散が個別になる場合、packed rowsの一時データは従来の9成分copyで上書きされる。

**23,499旧新比較一致**（23,067出力/状態/軌跡等、432既存入力棄却）。
原版のsource/pyd/全validator dependency SHAが同じことを検証して保存済み原版記録を参照し、同じ全validatorを新candidateで実行した。
Phi/受理式・距離/AFV/尤度/予測/query演算は原版のままで、全73 GPU SASS body/resourceも一致。copy-only2kernelを追加した。
独立整数Philoxによる祖先・全16状態bit/NaN payload/negative zero・common/imported covariance・端数/最大seed/4draw尾・queued切替の**361新規tests**は原版/新版で通過。
既存のimport/reinitialize/KF invalidation6casesを加えた**367 targeted tests**も原版/新版で通過した。
PF/BVH/signal等**2,920 regression tests PASS**。367casesのmemcheck0 errors/0bytes leaked、racecheck0 hazards、synccheck0 errors。

2,048/10万/100万粒子×10stageの**30条件・5round×20**を別のpersistent workerで順次・交互計測。
各timed log-weight/returned estimateとstage/round末stateがbit一致。
単独B60や未変更B0等のstage-batch時間が大きく変動したため、同じsource/pyd/helperを固定し、100万粒子4条件を**1 measured callずつ交互の5×20**で確認した。
追加400旧新pairのlog-weights/returns/末stateもすべてbit一致。

| 1call交互確認 / 100万粒子 | 直前 (ms) | packed gather (ms) | 中央値短縮 | 改善したround |
|:---|---:|---:|---:|---:|
| 共通共分散 B60 resample | 6.1303 | 4.6006 | 24.95% | 4/5 |
| native PR update step | 14.6844 | 13.0132 | 11.38% | 5/5 |
| 公開 PR update step | 12.5106 | 9.8135 | 21.56% | 5/5 |
| 未変更の単独 PR weight | 5.2302 | 5.4110 | -3.46% | 1/5 |

公開PR stepは5/5で約21.6%短縮し採用。個別共分散の公開PR+KFはstage-batchで41.8751→41.9246msで、改善対象ではない。
小粒子数へ一律の速度比は主張しない。公開PR stepのstage-batchは2,048粒子0.3165→0.3247ms、
10万粒子1.7979→1.7729msで、改善した組はそれぞれ2/5・3/5。
GPU/hostは非占有で、異なるprotocolの絶対時間・過去の速度比を組み合わせない。
元の共分散uniform保証・再サンプリング方式/反復数・ESS条件を保ち、測位計算の近似や粒子削減は行わない。
長時間実データの測位誤差分布は新規評価していない。

CUDA12.8 / MSVC Release / compute7.5 / C++20 / no fast math。
[全30条件・全round/sample・400交互pair・23,499比較・source/pyd/helper SHA・verification/archive](pf_packed_gather_gpu_20261013.json)、
[原版73body/resource一致・copy-only2kernel・正確なpatch](pf_packed_gather_device_code_20261013.json)、
[最終source/build/installed/検査整合](pf_packed_gather_completion_check_20261013.json)。
再現helperは `bench_pf_packed_gather_compare.py` と `bench_pf_packed_gather_paired_compare.py`。
前者は `--baseline` / `--candidate` / `--baseline-dir` / `--baseline-validation-report` / `--sizes 2048 100000 1000000` / `--rounds 5` / `--repeats 20` / `--output`。
原版記録は旧source/pydと全validator dependencyのSHA一致を必須とし、candidateは全fixtureを実行する。
paired helperは同じfull reportを `--validated-report` に渡す。各sampleのinitialize/restore・active context3warmup・IPC・hashは時間外、完了同期を含むwall timeを測る。
BVH mirror chunksとsignal epoch batchの採用source/build/installedも保持した。

### 採用しなかったblock proposal共有案

先に、同じblockのPhilox候補列をcooperating lanesがB1..256の同じabsolute offsetから一度だけ生成しsharedへ保存する案を試した。
own RNG・FP64差分・元のFP32受理式/順番は同じ。23,499比較一致、独立181draft testsも原版/試行で通過。
原版73 SASS/resourceを保持、shared proposal kernel38register/1KB sharedを追加。
30条件・5×20では100万粒子公開PR12.4705→12.4706ms、公開PR+KF41.9774→41.9595msで全体の利益がなく不採用。
GPU activityもselection1.9795→1.9784msで、重複計算の削減がこの条件の律速を改善しなかった。
この試行はarchive/patchへ保存して削除し、packed gatherは元の祖先選択kernelをそのまま使用する。
[不採用案の全計測・23,499記録・181tests・GPU区間・archive/patch](pf_shared_proposals_rejected_20261013.json)、
[device-code比較](pf_shared_proposals_rejected_device_code_20261013.json)。
再現helper `bench_pf_shared_proposals_compare.py` / `bench_pf_shared_proposals_recheck.py` も保存した。
不採用案には追加full regression/sanitizerを行わず、361testのpacked gather採用案で実施した。

## Undifferenced carrier AFV scalar-sharing and unroll trials (2026-10-13)

搬送波 AFV で粒子ごとに繰り返していた `1.0 / wavelength` と `1.0 / (sigma_cycles*sigma_cycles)` を、
既存 barrier 前に block 内の thread0 が同じ GPU FP64 式で計算し共有する案を試した。
次に、観測2件ずつの loop 展開を追加し、距離演算を進めやすくする案も検証した。
距離/sqrt・位相残差・AFV/rint・衛星順の加算・最後の prior 加算、入力検査、API、粒子/観測数、RNGを保つ。
ソース全体を原版から再構成し、差分がscalar保存位置と `#pragma unroll 2` のみに収まることを確認。
両案とも全73 GPU bodyのうち未変更72の SASS/resource は一致。AFV は42 register/0 stackのまま、static shared のみ16 byte追加。

**両案を不採用**。各案を2,048/10万/100万粒子・24条件・5交互round×20で測定。
逆数共有だけの案は100万粒子の公開 MUPF 中央値22.3039→23.7366ms、改善した組は3/5。
2件展開案の公開 MUPF は21.9035→21.7602ms、3/5に留まり、未変更controlも大きく変動した。
単独 AFV の12衛星13.0357→9.7542msという短縮もhost変動を含むため、その数字だけで採用しない。
同じ source/pyd/helper SHAを固定し、100万粒子の3条件を1 measured callずつ旧新版交互の5round×20で追加確認した:

| 1call交互確認 / 100万粒子 | 直前 (ms) | 共有+2件展開 (ms) | 中央値短縮 | 改善したround |
|:---|---:|---:|---:|---:|
| 単独 AFV / 12衛星 | 6.2938 | 6.1382 | 2.47% | 4/5 |
| 公開 PR+AFV MUPF step / 12衛星 | 21.6386 | 21.5297 | 0.50% | 5/5 |
| 未変更の公開 PR step | 14.6470 | 14.5266 | 0.82% | 3/5 |

公開 MUPF は5/5短縮したが差は約0.5%、単独 AFV は約2.5%。未変更 PR も約0.8%短縮し、他の衛星数・粒子数で利益は安定しない。
このhostで効果的な全体高速化を立証できなかったため、衛星数に応じた追加dispatchを作らず、採用済みPF実装へ戻した。
非占有hostであり、異なるprotocolの絶対時間や過去の速度比を組み合わせない。
公開 MUPF はpredict→PR update→AFV update→estimate、各updateのESS判定/条件付きresampleを含む。
PR/AFV間のresamplingや衛星subsetを保つため、今回の検証では両updateを一括化しない。

各案19,531旧新比較一致（19,099出力/状態等、432既存PR入力棄却）。
partial blocks、衛星0/1/4/12/33/64/65/128/1,228、波長/分散の変更・非有限値/underflow/overflow、
queued入力所有と既存予測/KF/DD/GMM/resample軌跡を照合した。
独立DD GPU計算、整数/半周期のrint、empty更新、IEEE挙動、最大shared/tail、queued scalar切替の57 draft testsも原版/共有案/共有+2件展開案すべてで通過。
全timed log-weight・returned estimate・各stage/round末stateをbit照合し、追加300旧新sample pairの出力も一致。

CUDA12.8 / MSVC Release / compute7.5 / C++20 / no fast math。
[全24+24条件・5×20の全sample・300交互pair・19,531比較・source/pyd/helper SHA・2patch/archive](pf_afv_constants_unroll_rejected_20261013.json)。
再現helperは `bench_pf_afv_constants_compare.py` / `bench_pf_afv_unroll_compare.py` / `bench_pf_afv_paired_compare.py`。
原版は直前の `pf_kf_wide_cache_gpu_20261013.json` のcandidate。対象patchを適用してbuildし、source/binary/helperを固定して比較する。
前2helperは `--baseline` / `--candidate` / `--baseline-dir` / `--sizes 2048 100000 1000000` / `--rounds 5` / `--repeats 20` / `--output`。
paired helperは2件展開の同一full reportを `--validated-report` に渡す。原版で実施済みの照合をSHA一致で参照し、同じworkerの3warmup後に1callを測定する。
同期を含むwall time。initialize/restore/IPC/hashは時間外。timed log-weights/returnsとstage末stateを照合する。
57testsは不採用trialのarchiveに保存して通常suiteへ追加せず、新規full regression/sanitizerは実行しない。
PF source/build/installedを直前のwide KF採用版にexact SHAで復元。BVH mirror chunksとsignal epoch batchの採用版も保持。
[最終復元・archive・公開表記の整合](pf_afv_constants_unroll_rejected_completion_check_20261013.json)。
長時間実データの測位誤差分布は新規評価していない。

## Signal input-packing allocation and diagnostic-name trials (2026-10-13)

signal bindingで、navigation参照vectorのdefault `py::array_t<int>` がチャネルごとに未使用の空NumPy配列を生成していた。
installed pybind11 `array_t()` の実装とbuild includeを確認し、default-null `py::object` slotで元のconverted arrayをmove保持する案を試した。
次に、正常値でも作っていた `channels[index].field` 診断名を、元のfinite/positive predicateが失敗した時だけ作る案を追加した。
型変換の回数/順番・検査内容・元validatorの例外型/文言・converted nav arrayのcall終了までの寿命を保つ。
全8 GPU SASS body/resourceが同一、CU/header/Python callerは変更しない。

**両案を不採用**。参照slotのみの633比較/20条件・3round×10の先行測定を別archiveへ固定した。
組合せ案は737旧新比較一致（IQ/metadata/変換657、native例外80）。list/float64/temporary `__array__` のnav変換、
非有限値・indexed field名・error precedenceを確認し、42 draft testsも旧新版双方で通過。
weakrefによる一時array解放、返却array所有、observable `__float__` 2/native・3/publicの変換回数も一致。

20条件・5交互stage-batch round×20では中央値の改善が見えたが、公開経路の改善pairは3〜4/5に留まった。
そこで同じfrozen workerを使い、1 measured callずつ旧新版を交互にする10条件・5round×20の確認を追加した。
32epoch×2,600 samplesの公開batch（末行は別sample数）:

| 1call交互確認の公開経路 | 直前 (ms) | 試行 (ms) | 改善したround |
|:---|---:|---:|---:|
| 固定nav bit / 12チャネル | 6.6880 | 7.3727 | 2/5 |
| 固定nav bit / 65チャネル | 30.2313 | 28.6781 | 5/5 |
| nav bit列あり / 12チャネル | 6.5255 | 5.5098 | 5/5 |
| nav bit列あり / 65チャネル | 27.7580 | 23.2179 | 5/5 |
| urban / 1,200三角形 | 30.5099 | 29.0162 | 3/5 |
| 長いIQ / 8epoch×260,000 samples / 12チャネル | 19.9402 | 19.7851 | 3/5 |

nativeとnav列ありの公開batch等で改善した一方、よく使う12チャネル固定navの公開batchが悪化し、urban改善も安定しない。
正常時のallocation減少や個別の良い条件だけで全体の採用を正当化せず、直前のsignal実装へ戻した。
非占有hostで、stage-batchとper-sample方式の絶対時間や過去の速度比を組み合わせない。
各測定は実入力validation/変換/H2D/kernel/完了同期/返却IQを含み、urbanはmetadata/LOS/反射/trajectory yieldも含む。
per-sample方式のinput構築・active-context warmup4call・IPC・hashは時間外。全timed IQ/metadataのbytesを照合する。
CUDA12.8 / MSVC Release / compute7.5 / C++20 / no fast math。
[全20 stage-batch/10 per-sample時間・全round/sample・737記録へのSHA・first variant・source/pyd/helper SHA・exact patches/archive](signal_input_packing_rejected_20261013.json)。
再現helperは `bench_signal_input_packing_compare.py` と `bench_signal_input_packing_paired_compare.py`。
各 `--baseline`/`--candidate`/`--baseline-dir`/`--stages`/`--rounds 5`/`--repeats 20`/`--output` を使う。
参照slotのみの測定helperもfirst archiveに固定し、後の検査名変更やhelper拡張と混ぜない。
42testsは不採用trialのarchiveに保存して通常test suiteへ追加せず、新規full regression/sanitizerは実行しない。
signal source/build/installedを直前batch採用版にexact SHAで復元。PF wide KFとBVH mirror chunksの採用版も保持した。
[最終復元・archive・公開表記の整合](signal_input_packing_rejected_completion_check_20261013.json)。
長時間実データの測位誤差分布は新規評価していない。

## Exact wide velocity KF geometry cache (2026-10-13)

33〜64衛星の速度KF更新でも、共通時計driftを取り除く平均計算と逐次KF更新で、
同じ粒子×衛星のFP64 range・LOS方向の二度の計算を省き、元のLOS方向と衛星速度補正済み観測を再利用する。
従来は32衛星を超えると2passともgeometryを再計算していた。33〜48はcache48、49〜64はcache64、128thread/block。
1〜32の既存cache16/32と65以上の元uncached/block256を保つ。
KF本体は元cached kernelのexact copyで、関数名とinteger validity maskのunsigned64/defined shiftだけを変更。
全FP64式・観測累積順・逐次velocity/covariance更新・symmetrization/clamp・尤度・RNG・粒子/観測数・API/既定値を保つ。

| 公開 predict＋PR＋速度KF＋ESS＋必要時B60 resample＋estimate | 直前 (ms) | 採用版 (ms) | 時間短縮 |
|:---|---:|---:|---:|
| 100,000粒子 / 33衛星 | 11.0965 | 8.9792 | 19.1% |
| 100,000粒子 / 48衛星 | 15.3994 | 12.3912 | 19.5% |
| 100,000粒子 / 49衛星 | 15.6487 | 12.5945 | 19.5% |
| 100,000粒子 / 64衛星 | 19.9346 | 15.9410 | 20.0% |
| 1,000,000粒子 / 33衛星 | 111.8946 | 89.3338 | 20.2% |
| 1,000,000粒子 / 48衛星 | 148.9486 | 125.1384 | 16.0% |
| 1,000,000粒子 / 49衛星 | 153.7518 | 127.2152 | 17.3% |
| 1,000,000粒子 / 64衛星 | 196.5619 | 161.9534 | 17.6% |

上表は **16.0〜20.2%短縮**。native/public33/48/49/64衛星×10万/100万粒子の16条件は全5交互roundで改善。
2,048/100,000/1,000,000粒子×13stageの39条件、5交互round×20回の全sampleを保存した。
nativeは入力/H2D/KF/完了sync、公開は実Python入力処理・predict・PR/KF・ESS・条件付きB60 resample・estimateを含む。
初期化・状態復元・IPC・出力hashを時間外に置き、各timed log-weight/返却値と各round末stateを旧新照合する。
非占有host、CUDA12.8 / MSVC Release / compute7.5 / C++20 / no fast math。
100万粒子の変更対象外対照: public_KF_sat32: 88.3733→88.5355ms、public_KF_sat65: 207.0464→206.8075ms、public_PR_step: 12.6140→12.6280ms。
RBPF velocity KFを使う上記観測数/粒子数の条件の測定であり、既定PRのみ・小粒子数・初回・全実データへ速度比を一般化しない。
過去の速度比と掛け合わせない。[全sample/round/入力・出力・source/pyd/helper SHA](pf_kf_wide_cache_gpu_20261013.json)。

旧新版18,941比較一致（18,509出力、従来PRの入力棄却432）。従来18,591に、33/47/48/49/63/64/65観測、
127/128/129 tails、masked/near/nonfinite/last-only/nonsymmetric covariance/invalid S、2回更新、queued predict/入力の直後上書き/resamplingを追加した。
新規58testsは65観測へのzero-weight paddingでretained uncached GPU kernelを独立oracleにし、元の観測順と有効観測数を保つ。
同じ58testsを直前binaryでも通してpadding/variance dispatchのoracleを確認した。bit63のみ有効、scalar NaN/Inf/tiny sigma、
共分散の非対称/非有限/負値、1m cutoff、ECEF、queued growth・pointer swap・返却array所有を含む。
関連回帰2,559tests通過。58testsのmemcheck/racecheck/synccheckはerrors/hazards/warnings/leaks0。
精度toleranceを緩めず、長時間実データの測位誤差分布は今回新しく計測していない。

既存71 GPU SASS body/resource全同一、新KF wide kernel2個。33〜64衛星のregisterは元uncached94→86、
per-thread stackは元0→cache48:1536/cache64:2048 bytes。ローカル領域増加を含めてcomplete callを測定した。
owner buffer容量/allocation/H2D回数/packed observation/shared9*nSat/launch数/stream ownershipは同じ。
BVHのearly reject試行は不採用としてexact復元し、parallel mirror chunksとsignal batchのsource/build/installedを保つ。
[GPU/source body照合・exact patch・独立oracle/log/archive](pf_kf_wide_cache_device_code_20261013.json)、
[最終source/build/installed・公開表記整合](pf_kf_wide_cache_completion_check_20261013.json)。

```powershell
$env:PYTHONPATH='python'
python benchmarks/bench_pf_kf_wide_cache_compare.py --baseline dist/pf_kf_wide_cache_baseline/_gnss_gpu_pf_device.cp312-win_amd64.pyd --candidate build/diag/Release/_gnss_gpu_pf_device.cp312-win_amd64.pyd --baseline-dir dist/pf_kf_wide_cache_baseline --sizes 2048 100000 1000000 --rounds 5 --repeats 20 --output dist/pf_kf_wide_cache_rerun.json
```

## Certified reflection numerator early-rejection trial (2026-10-13)

反射のMoller-Trumbore判定で、元と同じFP64分子を逆数より先に計算し、
finite determinantが元1e-12 cutoffを通る場合の確実なwide missだけで `1/a` を省く案を試した。
逆符号かつ|numerator|≥1なら最大全域の有限|a|でも商がsigned zeroへunderflowしない。
同符号かつ|numerator|/2>|a|なら逆数/積のFP64丸め後もu>1を保つ。
境界・非有限値は元の逆数/述語を使い、ray正規化・FP64式・三角形順・tie・閾値・候補数を保つ。
512以上のordered parallel pathに専用2kernelを追加し、既存10 GPU SASS body/resourceは全同一。
新旧cached kernelのregister78/77・stack0・shared5120 bytes、owner allocation/payloadも同じ。

**不採用**。旧新466 LOS/reflectionケースがbit一致し、64追加sceneではretained stateless GPUとも一致したが、
8,192三角形・129epoch×32衛星の公開traceは **36.3267→39.5976ms**、
3交互round全てで悪化した。既存geometry/候補を減らさず逆数を省く案でも、complete callで利益が出ない。
511/512境界・1e-7〜1e100 scale・両determinant符号・u0/1/2隣接値・signed zero・ECEF・mirror chunksを確認。
511/512/1200/8192三角形×1x16/1x192/129x32×4stageの48条件・3交互round×5回。
native multipath/public trace/cold owner初回trace+close/LOS対照を含み、全timed返却bitを照合。
非占有host、CUDA12.8 / MSVC Release / compute7.5 / C++20 / no fast math。
小規模の小差を普遍的な速度比へ一般化しない。拒否trialには新規回帰/sanitizerを実行せず、source/build/installedを直前採用版へexact復元した。
[全48時間・466比較・source/pyd/helper SHA・SASS/resource・exact patch/archive](bvh_early_intersection_rejected_20261013.json)。
再現helperは `bench_bvh_early_reject_compare.py`、直前parallel mirror chunksのsource5/binaryをbaseline-dirへarchiveし、
`--counts 511 512 1200 8192 --shapes 1x16 1x192 129x32 --rounds 3 --repeats 5` を使う。
長時間実データの測位誤差分布は新しく計測していない。

## Covariance component planes and paired signal noise trials (2026-10-13)

PF の速度共分散を particle-major 9 doubles から component-major 9 planes へ変更し、
予測・Doppler KFで隣接threadのload/storeを連続化する案を試した。公開Nx16状態形式・容量・APIは保持し、
全sourceが共分散index変更とcopy-only gather変更だけでexact CRLF再構成できる。
FP64算術式・RNG・衛星順・粒子/観測数を変えない。GPU71関数中62 SASS bodyが同一、共分散を扱う9関数が変更。

**不採用**。旧新版18,591比較が一致し、先行189ケースの入力・state/log-weight NPZもbit差0だったが、
予測やsystematic resampling単独の改善が、公開処理全体の安定した改善につながらなかった。
共分散が粒子ごとに異なるcoalesced/legacy resamplingにも悪化があり、配置変更を採用しない。

| 公開 predict＋PR＋任意のKF＋ESS＋必要時B60 resample＋estimate | 直前 (ms) | 試行 (ms) | 改善した交互round |
|:---|---:|---:|---:|
| 100,000粒子 / PR | 1.7946 | 1.7757 | 2/3 |
| 100,000粒子 / PR＋KF | 6.0637 | 6.4293 | 1/3 |
| 1,000,000粒子 / PR | 12.4636 | 12.4906 | 1/3 |
| 1,000,000粒子 / PR＋KF | 41.8795 | 46.6076 | 0/3 |

10万/100万粒子×14stageの28条件、3交互round×5回の探索測定。
nativeはcovariance prediction/KF/3 resampling、公開は実Python入力処理・同期・返却値を含む。
初期化・状態復元・IPC・hashを時間外に置き、各timed log-weight/返却値と各round末stateを照合した。
CUDA12.8 / MSVC Release / compute7.5 / C++20 / no fast math、非占有host。
小さな差や単独stageの差を普遍的な速度比と解釈せず、この試行の条件で採用価値がないと判断した。
[全28中央値・round・source/pyd/helper SHA・189診断・GPU資源・復元patchとarchive](pf_covariance_planes_rejected_20261013.json)。
全raw sample/18,591記録は同reportにSHAを記録したprivate測定JSONに保存。
比較helperは `bench_pf_covariance_planes_compare.py`。直前source5とbinaryをarchiveし、
`--baseline`/`--candidate`/`--baseline-dir`/`--sizes 100000 1000000`/`--rounds 3`/`--repeats 5`/`--output`で再現する。
不採用trialには新規回帰suite/sanitizerを実行せず、直前source/build/installed binaryをexact SHAで復元した。
長時間実データの測位誤差分布は新しく計測していない。

信号生成の2個の雑音を `curand_normal` 2回から `curand_normal2` 1回へ変える案も試した。
ビルド後の**全8 GPU SASS body/resourceが完全同一**で、コンパイラが既に同じ処理へ整理していた。
有効なdevice変更がないため、不採用。timing/出力比較/回帰/sanitizerは行わず、元source/binaryをexact復元した。
[静的照合・source/binary SHA・復元patchとarchive](signal_noise_pair_rejected_20261013.json)。
[PF/BVH/Signalの最終source/build/installed整合・archive・公開表記確認](optimization_covariance_noise_rejected_completion_check_20261013.json)。

## Original Doppler geometry with64-bit validity for33–64 observations (2026-10-13)

33〜64衛星のDoppler更新で残っていた、時計drift・尤度・WLS間の粒子別geometry再計算を省いた。
33〜48衛星にcache48、49〜64衛星にcache64を使い、最初のFP64方向3成分を後続loopで再利用する。
32bit flagを流用せず、専用templateのvalidity maskをunsigned64bitとし、bit63まで定義されたshiftで扱う。
ソース検証で、専用bodyは元のcached bodyから**関数名と整数mask幅だけが異なり、全FP64式は完全同一**と確認した。
block128、元のrange<1/NaN semanticsと観測の加算順・solver・gain/clippingを保持する。
1〜32衛星の既存cache8/16/24/32と65衛星以上の元uncached/block256、API・既定値・粒子/観測数・RNGを保つ。

| 公開 predict＋PR＋Doppler＋ESS＋必要時B60 resample＋estimate / gain0.25 | 直前 (ms) | 採用版 (ms) | 時間短縮 |
|:---|---:|---:|---:|
| 100,000粒子 / 33衛星 | 8.8541 | 5.3775 | 39.3% |
| 100,000粒子 / 48衛星 | 12.2061 | 7.1357 | 41.5% |
| 100,000粒子 / 49衛星 | 12.4175 | 7.2563 | 41.6% |
| 100,000粒子 / 64衛星 | 15.7485 | 9.0152 | 42.8% |
| 1,000,000粒子 / 33衛星 | 85.6612 | 53.2639 | 37.8% |
| 1,000,000粒子 / 48衛星 | 117.8130 | 72.0829 | 38.8% |
| 1,000,000粒子 / 49衛星 | 120.7339 | 74.5985 | 38.2% |
| 1,000,000粒子 / 64衛星 | 153.6282 | 91.6412 | 40.3% |

上表の公開処理は**37.8〜42.8%短縮**。
100,000/1,000,000粒子のnative/公開33/48/49/64衛星×gain0/0.25の**対象32 timed条件は全5交互roundで改善**。
2,048/100,000/1,000,000粒子×21stageの**63条件・5交互round×20回**を保存。
公開処理にはPython入力準備・予測・PR/Doppler・ESS・必要時resampling・estimateと完了syncを含む。
初期化・復元・IPC・出力hashは時間外。CUDA12.8 / MSVC Release / compute7.5 / C++20 / no fast math。
非占有hostの対照例（100万粒子）: public_doppler_sat32_gain0.25: 53.5825→53.8040ms、public_doppler_sat65_gain0.25: 158.6166→157.9943ms、public_PR_control: 12.6399→12.6342ms。
対照や小規模の変動、初回owner生成・長時間実データ全体へ速度比を一般化せず、過去の比率と掛け合わせない。
[全sample・round中央値・入力/出力・source/pyd/helper SHA](pf_doppler_wide_cache_gpu_20261013.json)。

旧新版**15,189比較が一致**（14,757出力bits、従来PRの入力棄却432）。
直前14,202条件に、33/47/48/49/63/64/65衛星と127/128/129等のparticle tails、normal/masked/near/nonfinite/last-only、
gain0/0.25/1.5/NaN・2回更新、queued predict/update・観測数変更・caller入力の直後上書き・resamplingを加えた。
全timed state/log-weight/返却値も一致。324先行診断は全inputと旧新版state/log-weightをNPZで固定、bit差0。
追加75testsと既存160testsの**235 tests**は、65観測へのpaddingでretained uncached GPU kernelを独立oracleにする。
追加衛星を各粒子から1m未満に置き、元のrange cutoffで算術前に除外して元の観測順を保つ。
既存2test fileのoracle padding/dispatchを33→65へ更新したexact patchも保存し、新cacheをoracleにしない。
48/49と64/65、64bit maskのbit63のみ有効、scalar非有限値・signed zero・tiny sigma・ECEF、
予測後の再計算・buffer growth・返却array所有・元の768衛星shared容量境界を含む。
関連回帰**2,501 tests通過**。235testsのmemcheck/racecheck/synccheckはerrors/hazards/warnings/leaks0。
精度toleranceを緩めず、候補/観測/粒子を削減していない。長時間実データの測位誤差分布は今回新しく計測していない。

**既存69 GPU関数のSASS body/resourceが全同一、新wide-cache関数2個**。
33〜64衛星のregisterは元uncached90→新76、per-thread stackは元160→cache48:1312/cache64:1696 bytes。
作業領域増加を含めて実測した。owner observation payload・allocation/H2D回数・shared8*nSat doubles・launch回数・stream ownershipは同じ。
他のPF更新、BVH parallel mirror chunks、signal batchのsource/build/installed SHAを保持。
[GPU命令/resource・source/oracle再構成patch・324診断と採用archive](pf_doppler_wide_cache_device_code_20261013.json)、
[最終source/build/installed・検証log・公開表記の整合](pf_doppler_wide_cache_completion_check_20261013.json)。

再現時は直前cache8/16/24/32版のbinary/sourceを保存する。helperがbaseline_referenceのSHAを照合する。

```powershell
$env:PYTHONPATH='python'
python benchmarks/bench_pf_doppler_wide_cache_compare.py --baseline dist/pf_doppler_wide_cache_baseline/_gnss_gpu_pf_device.cp312-win_amd64.pyd --candidate build/diag/Release/_gnss_gpu_pf_device.cp312-win_amd64.pyd --baseline-dir dist/pf_doppler_wide_cache_baseline --sizes 2048 100000 1000000 --rounds 5 --repeats 20 --output dist/pf_doppler_wide_cache_rerun.json
```

## Reuse original Doppler geometry for17–32 observations (2026-10-13)

multi-GNSS等で17〜32衛星を扱うDoppler更新にも、元のFP64距離・LOS方向の再利用を適用した。
以前は16衛星を超えると、時計drift・尤度・WLSの各loopで同じ粒子×衛星geometryを最大3回計算していた。
17〜24衛星はcache24、25〜32衛星はcache32、128thread/blockで最初のFP64方向3成分を再利用する。
既存のtemplate本体は一切変えず、host dispatchに2個の既存式のinstantiationを追加した。
32bitのvalidity maskを上限まで使い、33衛星以上は元の256thread/kernel、1〜16衛星は直前のcache8/cache16を保持する。
粒子数・観測数・既定gain0.25・API・モデル・FP64算術を保つ。

| 公開 predict＋PR＋Doppler＋ESS＋必要時B60 resample＋estimate / gain0.25 | 直前 (ms) | 採用版 (ms) | 時間短縮 |
|:---|---:|---:|---:|
| 100,000粒子 / 17衛星 | 5.4587 | 3.6435 | 33.3% |
| 100,000粒子 / 24衛星 | 7.0468 | 4.5257 | 35.8% |
| 100,000粒子 / 25衛星 | 7.1544 | 4.5124 | 36.9% |
| 100,000粒子 / 32衛星 | 9.5683 | 6.5924 | 31.1% |
| 1,000,000粒子 / 17衛星 | 51.3792 | 34.6996 | 32.5% |
| 1,000,000粒子 / 24衛星 | 66.3500 | 42.7336 | 35.6% |
| 1,000,000粒子 / 25衛星 | 68.4860 | 43.8832 | 35.9% |
| 1,000,000粒子 / 32衛星 | 83.5292 | 52.1874 | 37.5% |

上表の公開処理は**31.1〜37.5%短縮**。
100,000/1,000,000粒子のnative/公開17/24/25/32衛星×gain0/0.25の**対象32 timed条件は全5交互roundで改善**した。
2,048/100,000/1,000,000粒子×21stageの**63条件・5交互round×20回**を保存。
nativeは観測準備/H2D/kernel/完了sync、公開処理はPython入力準備・予測・PR/Doppler・ESS・必要時resampling・estimateを含む。
初期化・状態復元・IPC・出力hashは時間外。CUDA12.8 / MSVC Release / compute7.5 / C++20 / no fast math。
非占有hostの対照例（100万粒子）: public_doppler_sat16_gain0.25: 33.6492→33.6690ms、public_doppler_sat33_gain0.25: 85.7689→86.4656ms、public_PR_control: 12.5466→12.5558ms。
小粒子数・変更対象外・初回owner生成や長時間実データ全体へ速度比を一般化せず、過去の比率と掛け合わせない。
[全sample・round中央値・入力/出力・source/pyd/helper SHA](pf_doppler_large_cache_gpu_20261013.json)。

旧新版**14,202比較が一致**（13,770出力bits、従来PRの入力棄却432）。
直前13,215条件に、127/128/129等のparticle tails、17/23/24/25/31/32/33衛星、normal/masked/near/nonfinite/last-only、
gain0/0.25/1.5/NaN・2回更新、queued predict/update・観測数変更・入力の直後上書き・resamplingを加えた。
324先行診断は全入力と旧新版state/log-weightのNPZを固定し、bit差0。全timed state/log-weight/返却値も一致。
追加75testsと既存85testsの**160 tests**では、33観測へpaddingして追加衛星を1m未満で算術前に除外し、
retained uncached GPU kernelを独立oracleにしてbit照合する。
以前の17paddingを33へ更新したexact test patchも保存し、oracleを新cacheへ切り替えてしまうことを防いだ。
24/25と32/33の境界、maskのbit31のみ有効なケース、FP64 scalar非有限値、1m cutoff、signed zero、tiny sigma、ECEF、
predict後の再計算、buffer growth、caller入力/返却array所有、元の768衛星shared容量境界を確認。
関連回帰**2,426 tests通過**。160testsのmemcheck/racecheck/synccheckはerrors/hazards/warnings/leaks0。
精度toleranceを緩めず、候補/観測/粒子削減をしていない。長時間実データの測位誤差分布は新しく計測していない。

**既存67 GPU関数の完全なSASS bodyとresourceが全同一、新cache関数2個**。
17〜32衛星のregisterは旧90→新70、per-thread stackは旧160→cache24:736/cache32:928 bytes。
この作業領域増加を含めて実測した。owner payload・allocation/H2D回数・shared8*nSat doubles・launch回数・stream ownershipは同じ。
packed satellite、その他PF更新、BVH parallel mirror chunks、signal batchのsource/build/installedを保持。
[GPU命令/resource・source/oracle再構成patch・採用archive/324診断 SHA](pf_doppler_large_cache_device_code_20261013.json)、
[最終source/build/installed・検証log・公開表記の整合](pf_doppler_large_cache_completion_check_20261013.json)。

先に試した**時計driftとWLS matrixのloop統合は不採用**。
216先行診断と13,215比較でbit一致し、従来67 SASS/resourceも保持したが、新関数はregister70→94。
3round×5の主な100万粒子publicは12衛星28.5037→27.7128ms、16衛星33.1233→32.1004msと約2〜3%短縮にとどまった。
変更していないlikelihood-only対照の変動がそれより大きく、今回の優先度として採用を見送った。
fused試案のexact source/pyd/helper/raw/patchと216件の旧新版input/outputを保存し、regression/sanitizersは試案に実行していない。
[統合案の棄却根拠・再現archive](pf_doppler_wls_fusion_rejected_20261013.json)。

再現時は直前cache8/cache16版のbinary/sourceを保存する。helperがbaseline_referenceのSHAを照合する。

```powershell
$env:PYTHONPATH='python'
python benchmarks/bench_pf_doppler_large_cache_compare.py --baseline dist/pf_doppler_large_cache_baseline/_gnss_gpu_pf_device.cp312-win_amd64.pyd --candidate build/diag/Release/_gnss_gpu_pf_device.cp312-win_amd64.pyd --baseline-dir dist/pf_doppler_large_cache_baseline --sizes 2048 100000 1000000 --rounds 5 --repeats 20 --output dist/pf_doppler_large_cache_rerun.json
```

## Reuse per-particle Doppler geometry across likelihood and WLS (2026-10-13)

PFの通常Doppler更新で、粒子×衛星の距離とLOS方向を最大3回計算する重複を省いた。
時計drift用の最初のloopで元のFP64方向3成分とrange cutoff結果を保存し、
尤度とWLS velocity更新のloopで再利用する。観測の加算順・4x4 solver・gain・速度補正上限を維持する。
1〜8衛星はcache8、9〜16衛星はcache16、128thread/block。17衛星以上は元の256thread/kernelを保つ。
公開API・既定gain0.25・粒子数・観測数・乱数・観測モデルを変えていない。

| 公開 predict＋PR＋Doppler＋ESS＋必要時B60 resample＋estimate / gain0.25 | 直前 (ms) | 採用版 (ms) | 時間短縮 |
|:---|---:|---:|---:|
| 100,000粒子 / 8衛星 | 3.4005 | 2.5332 | 25.5% |
| 100,000粒子 / 12衛星 | 4.2549 | 2.9910 | 29.7% |
| 100,000粒子 / 16衛星 | 5.3090 | 3.4831 | 34.4% |
| 1,000,000粒子 / 8衛星 | 32.0242 | 24.2402 | 24.3% |
| 1,000,000粒子 / 12衛星 | 40.7091 | 28.9450 | 28.9% |
| 1,000,000粒子 / 16衛星 | 55.2839 | 39.2947 | 28.9% |

100,000/1,000,000粒子のnative4/8/12/16衛星×gain0/0.25、および公開8/12/16衛星×gain0/0.25の
**対象28 timed条件は全5交互roundで改善**。公開処理全体の利得で採用する。
2,048/100,000/1,000,000粒子×19stageの**57条件・5交互round×20回**を保存。
初期化・復元・IPC・出力hashは時間外、更新と完了syncは時間内。
公開処理はPython入力準備・予測・PR/Doppler更新・ESS・条件付きresampling・最終推定を含む。
初回owner生成の改善や、長時間実データ全体の速度比としては扱わない。
17/32衛星のDopplerと公開PR-onlyは対照。小粒子数と非占有hostの時間変動を一律の利得としない。
変更対象外の100万粒子PR-onlyは13.3853→14.0185ms（+4.7%）だった。
2,048粒子・12衛星・gain0の公開処理も0.5041→0.5213ms（+3.4%）で、小規模の全条件改善とは扱わない。
[全sample・各round・入力/出力hash・source/pyd/helper SHA](pf_doppler_weight_cache_gpu_20261013.json)。

旧新版**13,215比較が一致**（12,783出力bits、従来PRの入力棄却432）。
既存10,296条件に、Dopplerの衛星数0/1/3/4/7/8/9/12/15/16/17/32/768、odd particle tails、
2回更新、gain0/0.25/1.5/NaN、zero/negative/nonfinite weights、1m range境界、tiny sigma、極端座標、signed zeroを加えた。
queueの観測数変更・入力の直後上書き・予測/resamplingも照合。全timed state/log-weight/返却値も一致。
追加**85 tests**は、retained uncached GPU kernelを独立oracleに使い**bit一致**を確認する。
17観測にpaddingし、追加衛星を各粒子から1m未満に置いて元のrange cutoffで算術前に除外するため、元観測の加算順を保てる。
通常/likelihood-only・WLS・scalar非有限値・ECEF・buffer growth・返却array所有・従来768衛星のshared容量境界を含む。
関連回帰**2,351 tests通過**。85testsのmemcheck/racecheck/synccheckはerrors/hazards/warnings/leaks0。
精度toleranceは緩めていない。長時間実データの測位誤差分布は今回新規計測していない。

**既存65 GPU関数のSASS命令body/resource数が全同一、新cache関数2個**。
registerは旧90→新70、per-thread stackは旧160→cache8:352/cache16:544 bytes。
cache保存の作業領域が増えるtradeoffを含めて実測した。hardware countersやoccupancyの改善を測定したという意味ではない。
owner観測payload・allocation/H2D回数・shared8*nSat doubles・kernel launch回数は同じ。
既存packed satellite、stream ownership、他のPF更新、直前BVH mirror chunksとsignal batchをsource/build/installed SHAで保持。
[GPU命令・resource・exact source再構成patch・採用archive](pf_doppler_weight_cache_device_code_20261013.json)、
[最終source/build/installed・検証log・公開表記の整合](pf_doppler_weight_cache_completion_check_20261013.json)。

予備比較の13,215条件と14timed条件（3round×5）は、完全なhelperをprivate archiveに固定した。
終了後、同helperの4箇所の1行suiteを展開し、重複importを除いてRuffを通した。
本計測のsource/pydは予備計測と同一で、両helperのSHAをmanifestに記録している。
予備計測の100,000粒子・8衛星publicには逆転があったため、本計測を増やして全5組の改善を確認した。

再現は、直前のpacked版binary/sourceを保存した上で行う。helperがbaseline_referenceのSHAを照合する。

```powershell
$env:PYTHONPATH='python'
python benchmarks/bench_pf_doppler_weight_cache_compare.py --baseline dist/pf_doppler_weight_cache_baseline/_gnss_gpu_pf_device.cp312-win_amd64.pyd --candidate build/diag/Release/_gnss_gpu_pf_device.cp312-win_amd64.pyd --baseline-dir dist/pf_doppler_weight_cache_baseline --sizes 2048 100000 1000000 --rounds 5 --repeats 20 --output dist/pf_doppler_weight_cache_rerun.json
```

## Parallel reflection mirrors remain reusable beyond the epoch budget (2026-10-13)

resident BVHの並列候補評価で、32MiBを超えるepoch群にも衛星間の共通反射計算を再利用する。
以前は全epochのmirrorが上限に収まらないと共有を無効にし、各衛星の候補評価で同じ計算を繰り返していた。
今回は収まるepoch数ずつmirrorを準備し、同じdefault streamで候補評価が完了してからbufferを再利用する。
partial winnerの書き込み位置をray数×tile数でずらし、最後に全rayのwinnerを一度だけ確定する。
対象は元のparallel pathの8,192 visits以上・16衛星以上・16,384 rays以下。
1epochが上限に収まらない場合は従来のnormal cache評価を保つ。

| 公開LOS＋反射 / 32衛星 | 直前 (ms) | 採用版 (ms) | 時間短縮 |
|:---|---:|---:|---:|
| 8,192三角形 / 257epoch / 8,224 rays | 81.2450 | 73.2454 | 9.8% |
| 12,000三角形 / 129epoch / 4,128 rays | 60.6382 | 55.0109 | 9.3% |
| 12,000三角形 / 257epoch / 8,224 rays | 122.3819 | 111.4445 | 8.9% |

native owner生成＋初回trace＋closeも、同じ3条件で**6.6〜7.6%短縮**。
native warm reflection・公開warm trace・初回trace/closeの**対象9条件は全5交互roundで改善**した。
3mesh×3batch×4stageの36 timed条件と9 memory条件を、5交互round×20回で測定。
公開処理には入力検証・H2D・LOS/全反射候補・同期・返却arrayを含む。
CPU BVH構築・IPC・出力hashは時間外。CUDA12.8 / MSVC Release / compute7.5 / C++20 / no fast math。
非占有hostでの計測で、変更対象外の8,191三角形・16 raysの初回処理は1.5078→1.5821ms（+4.9%、+0.0743ms）だった。
対照経路の変動や、mirrorを既に共有できる小batchを一律の高速化とは扱わない。
[全sample・round中央値・402条件の出力hash・source/pyd/helper SHA](bvh_parallel_mirror_chunks_gpu_20261013.json)。

**402条件でLOS/反射点/遅延がbit一致**し、全timed出力も一致。
追加62testsは独立したstateless GPU実装をoracleにし、8,191/8,192、15/16衛星、容量直前/直後/2chunk tail、
300,000/600,000 visitsの複数tile offset、1,398,102 visitsの1epoch上限超過、極端座標・法線cutoff・signed zero・nonfinite、
positive等遅延のright-first順、ECEF、serial切替、receiver更新・返却array所有を確認した。
関連回帰**2,266 tests通過**。62testsのmemcheck/racecheck/synccheckはerrors/hazards/warnings/leaksすべて0。
**既存10 GPU関数の完全なSASS命令bodyとresource数が全同一、新関数0**。
元のFP64算術、全候補、訪問順・tie、sp<62、gateを保ち、精度toleranceを緩めていない。
長時間実データの測位誤差分布を今回新たに評価したという意味ではない。

追加mirror容量は最大32MiB。実測対象では33,423,360 bytes（8,192 visits）または33,408,000 bytes（12,000 visits）。
全epoch数に比例して増やさず、owner内で容量を再利用し、closeで解放する。LOS-onlyはlazy、winnerは従来の最大640KiB。
PF packedとsignal batchはsource/build/installedとも直前SHAを保持。
以前のcached serialで不採用にしたepoch分割とは異なる並列候補経路の改善で、過去の速度比と掛け合わせない。
[GPU命令・resource・source/test再構成patch・採用archive SHA](bvh_parallel_mirror_chunks_device_code_20261013.json)、
[最終source/build/installed・検証log・公開表記の整合](bvh_parallel_mirror_chunks_completion_check_20261013.json)。

同時に候補評価の配置3案（64thread block、64winner shared、2ray/block）も比較した。
いずれも353条件でbit一致したが、12,000三角形の公開大batchで安定した利得を確認できず不採用。
2ray/blockは256 raysの公開traceが4.2613→4.5848msへ悪化した。
各36 timed/9 memory・3交互round×5のrawとexact source/pyd/helper/patchを保存し、試案のregression/sanitizersは実行していない。
[3案の棄却根拠と再現archive](bvh_candidate_layouts_rejected_20261013.json)。

## Vertex-layout trials and a measured candidate-kernel bottleneck (2026-10-13)

resident BVHの候補評価で、immutable頂点の配置と読み出しを2通り試した。
**両方不採用**とし、floor64のsource/build/installedへSHAで正確に復帰した。
PF packedとsignal batchも保持する。

**座標plane形式**はGPUで頂点値をそのままコピーし、同じ候補式へ読み込む案。
FP64、元の頂点値・訪問順・全候補・gateを保つソースでも、新consumerの出力が完全には一致しなかった。
最初の厳密検証で止め、同じ15入力の診断では**8条件で反射点/遅延にbit差**を確認した。
例として8,192三角形・2epoch×15衛星では反射点7成分と遅延1値が異なった。
17衛星やmirrorの32MiB境界でも差がある。差の小ささを理由にtoleranceを緩めず、不採用にした。
元の10 GPU関数はSASS/resource同一だが、新しい3関数のconsumerで一致を失った。
同じFP64式・命令種別数だけでは精度同値の根拠にならない。差の原因を特定したとは扱わない。
[診断raw・不一致input/output・source/pyd/helper・再構成patch](bvh_vertex_planes_precision_rejected_20261013.json)。
性能計測・追加regression/sanitizersは実行していない。

**Triangleをそのまま訪問順にコピー**する案では、local vertex再構成を避けて元candidate bodyを保った。
GPUコピーは一度だけ、追加72 bytes/visit、lazy allocationで上限8MiB。
8,192 visits未満、上限を超えるmesh、16,384 raysを超えるcached serialは元のpathを維持する。
旧新版**308条件でLOS/反射点/遅延がbit一致**。極端値、法線cutoff、positive tie、
8,191/8,192/8,193、mirror容量境界、116,508/116,509 visitsの8MiB境界、nonfiniteと呼び出し後の所有を含む。
全timed出力も一致。元10 SASS/resourceを保持、新3関数を追加。
しかし12,000三角形・4,128 raysの公開traceは**59.4905→59.4978ms**とほぼ変わらず、
同mesh・16 raysの初回owner生成/trace/closeは**2.9011→3.6725ms**へ悪化した。
36 timed条件・9 memory条件を3交互round×5で保存。非占有hostの時間変動も含め、追加メモリを正当化できず見送った。
[全raw・source/pyd/helper・resource/SASS・再構成patch](bvh_ordered_triangles_rejected_20261013.json)。
不採用後の追加regression/sanitizersは実行していない。

復帰したproduction版の12,000三角形・129epoch×32衛星を**nvprofのGPU activity timeline**で計測した。
warm native traceの内訳は次のとおり。2回のtrace出力hashも一致した。

| GPU activity | profiled time (ms) |
|:---|---:|
| 候補評価kernel（normal cache、mirror無効） | 72.626961 |
| LOS kernel | 0.965870 |
| winner finalize kernel | 0.002144 |
| 2 H2D＋3 D2HのGPUコピー合計 | 0.023102 |

候補評価は**GPU kernel時間の98.68%**、コピーも含むGPU activityの
**98.65%**を占めた。
この条件では、転送/取得の小さな削減より候補評価本体を優先する根拠になる。
これはprofiler下の1実行であり、通常wall timeの速度比や全都市/PFの比率へ読み替えない。
APIのcudaDeviceSynchronize時間はkernel実行と重なるため加算していない。
CPU build/出力hashは対象外。最初のtraceがlazy normalsを準備し、2回目をwarmとして抽出した。
[公開用にDevice列を省いたtimelineとsource/pyd/helper/raw SHA](bvh_floor64_kernel_profile_20261013.json)。

Nsight Compute 2025.1は--list-sections自体が0xc0000409で終了し、ハードウェアカウンターは未取得。
nvprofは初回0xc0000135、既に入っているtoolkit CUPTI DLL directoryを**計測子processのPATHにだけ**追加すると実行できた。
driver/toolの再installや永続環境の変更は行っていない。
timelineだけではFP64 pipeline飽和、memory stall、occupancy、命令latencyの区別はできない。
候補評価が支配することを根拠にし、律速要因の断定は保留する。
長時間実データの測位誤差分布は今回新しく測定していない。
[復帰・精度棄却・raw/SASS/patch・timelineの最終整合確認](optimization_layout_profile_completion_check_20261013.json)。

## Synchronization trials retain outputs but fail the performance gate (2026-10-13)

三角形数の下限調整から離れ、PF/BVHのhost待機と転送依存を実production pydで比較した。
下記の2試案は出力を保持したが、処理全体で安定した利得を確認できず**両方不採用**。
PF packedとBVH floor64のsource/build/installedへSHAで正確に復帰し、signal batchも保持した。

| 条件 | 直前 (ms) | 試案 (ms) | 時間増減 |
|:---|---:|---:|---:|
| PF 100,000粒子 / 公開PR predict/update/ESS/resample/estimate | 1.1469 | 1.1916 | +3.9% |
| PF 100,000粒子 / 公開PR+AFV MUPF step | 2.7346 | 2.7543 | +0.7% |
| PF 100,000粒子 / 4回predict+DD joint+最終estimate | 5.1391 | 5.5301 | +7.6% |
| PF 100,000粒子 / 4回predict+KF+最終estimate | 12.2355 | 12.1365 | -0.8% |
| BVH 12三角形 / 公開LOS+反射 / 16 rays | 0.1897 | 0.2138 | +12.8% |
| BVH 64三角形 / 公開LOS+反射 / 16 rays | 0.1749 | 0.1601 | -8.5% |
| BVH 64三角形 / 公開LOS+反射 / 4,128 rays | 0.7046 | 0.7393 | +4.9% |
| BVH 1,200三角形 / native LOS / 16 rays | 0.1256 | 0.1366 | +8.7% |

**PF pinned-source copy events**: PR/GMM/undifferenced AFVとDD/Doppler scratchのH2D直後に
timing-disabled eventを記録し、host source再利用ではその完了だけを待つ。
後続device copy/weight/predictは同じstreamで順序を保ち、buffer growthとdestroyはfull stream同期を残した。
追加event操作のある版で、100,000粒子の4回DD joint queueは全3組で遅くなり、公開PR/MUPF等も安定改善しなかった。
KF queueの0.8%短縮など一部の小幅改善を、PF全体の高速化として採用しない。
2,048/100,000粒子×8 stageの16条件を3交互round×5で測定。
公開stepは予測・観測更新・ESS・条件付きresampling・estimateまで含み、queueは4回予測/更新と最終estimateを含む。
queueのCPU observation準備も時間内。初期化・IPC・出力検査は時間外で、native syncで完了を確認する。
旧新版**11,895比較が一致**（既存PRの入力棄却432を含む）。追加Dopplerの1,566条件と33 queued列は、
3粒子数×11 modeで10回の予測/変化する観測を途中D2Hなしで投入し、caller入力を直後に変更してから
最終particle state・log weights・estimateを照合した。alias/growth/reuseと両scratch系の切替も含む。
**旧65 GPU関数のSASS/resource全同一**。[全raw・source/pyd/frozen helper・patch](pf_copy_event_rejected_20261013.json)。
初回helperの重複importは計測終了後、凍結版をarchiveしてから除去し、両版のSHAと変更内容を記録した。

**BVH resident wait removal**: residentの4箇所で、blocking D2H直前のdevice-wide waitを除去。
[CUDA12.8の仕様](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/api-sync-behavior.html)上、
同期D2Hはdefault streamでコピー完了後に返るため、結果取得の完了は保てる。
実測では12三角形の公開16 raysや1,200三角形のLOSが全3組で遅くなり、64三角形の小batch改善も
4,128 raysに広げると悪化した。待機数だけを性能指標にせず、元の待機を維持する。
12/64/1,200/12,000三角形×16/4,128 rays×4 stageの32 timed条件・8 memory条件を3交互round×8で測定。
公開warmは入力検査/H2D/全kernel/同期完了/D2H/返却配列、coldはowner生成・初回準備・closeも含む。
CPU BVH構築と出力hashは時間外。同processの旧新を各call/roundで交互にした。
**292条件でLOS/反射点/遅延がbit一致**、全timed出力も一致し、device memory増分0。
**旧10 GPU関数のSASS/resource全同一**。[全raw・source/pyd/helper・patch](bvh_copy_sync_rejected_20261013.json)。

hostは非占有で時間変動も記録する。待機方法によるdriver挙動やoccupancyを今回測定したとは扱わない。
FP64・候補数・粒子数・乱数系列・算術順・gate・物理モデルを削減しない。
性能gateで見送ったため、試案の追加regression/sanitizersは実行していない。
復帰先は直前の検証済みsource/pydそのもの。今回新しい長時間実データの測位誤差分布は測定していない。
[復帰・raw・source/pyd/SASS/patch・公開表記の整合確認](optimization_sync_rejected_completion_check_20261013.json)。

## Tiny resident meshes reuse original normals and ordered candidates (2026-10-13)

参照三角形64〜127でも、既存のGPU法線と元の訪問順を再利用する。
hostの適用下限2箇所を**128→64**へ変更した。全候補と計算式は維持する。
公開LOS+反射全体は、16 raysで**64〜79%**、4,128 raysで**33〜43%**、16,384 raysで**25〜32%**短縮。
127三角形・1epoch×16衛星は**0.6532→0.1386ms**、
129epoch×32衛星は**1.8813→1.0764ms**。
対象3メッシュ×4 batch×3 stageの36条件すべてで、5交互round×20の全5組が改善した。
初回owner生成・trace・closeも改善するが、64三角形・上限16,384 raysの短縮は3.7%に留まる。

| 条件 | 直前 (ms) | 新 (ms) | 時間短縮 |
|:---|---:|---:|---:|
| 64三角形 / 公開1epoch×16衛星 | 0.3834 | 0.1370 | 64.3% |
| 96三角形 / 公開1epoch×16衛星 | 0.5366 | 0.1476 | 72.5% |
| 127三角形 / 公開1epoch×16衛星 | 0.6532 | 0.1386 | 78.8% |
| 127三角形 / 公開1epoch×1 ray | 0.6353 | 0.1215 | 80.9% |
| 64三角形 / 公開129epoch×32衛星 | 1.1029 | 0.7367 | 33.2% |
| 127三角形 / 公開129epoch×32衛星 | 1.8813 | 1.0764 | 42.8% |
| 127三角形 / 上限512epoch×32衛星 | 5.2342 | 3.5360 | 32.4% |
| 127三角形 / 初回生成・trace・close / 1epoch×16衛星 | 0.7613 | 0.2850 | 62.6% |
| 64三角形 / 初回生成・trace・close / 上限512epoch×32衛星 | 3.1387 | 3.0211 | 3.7% |
| 63三角形 / 未変更公開1epoch×16衛星 | 0.3760 | 0.3711 | 1.3% |
| 128三角形 / 未変更公開129epoch×32衛星 | 0.9969 | 0.9835 | 1.3% |
| 127三角形 / 未変更LOS129epoch×32衛星 | 0.1867 | 0.1814 | 2.9% |

公開warmは入力検査・H2D・LOS/反射全kernel・同期・D2H・返却配列を含む。
native coldは構築済みCPU nodes/trianglesからowner生成・upload・初回準備・trace・closeを含む。
CPU BVH構築と出力hashは時間外。旧新production pydを同processで順次実行し、call/roundごとに交互にした。
全timed出力もbit hash照合した。[80 timed条件・20メモリ条件のraw](bvh_tiny_order_gpu_20261013.json)。
hostは非占有。63/128三角形とLOSは未変更controlとして測定した。
この小メッシュの改善を大型都市/PF全体へ一般化せず、過去の速度比と掛け合わせない。

**旧10 GPU関数のSASS instruction bodyとresource countsは全同一**。
FP64のnormal/mirror/交差/反射点/正遅延、元のright-first訪問ordinal/tie、sp<62とnonfinite zerosを維持。
候補数・ray数・丸め・物理モデルの削減はない。独立oracleとして使うstateless APIの下限512を維持した。
captureできないshared/overlapping leaf等は元のfallbackを使う。
新規対象のorder4＋normal32 bytes/visitはlazy allocation、winnerは40 bytes/ray、最大640KiB。
既存tile数、mirror条件と32MiB上限、16,384 rays超のcached serial、容量再利用とclose/releaseを維持。
LOS-onlyは反射cacheを作らない。[source/pyd/SASS/resourceと再構成patch](bvh_tiny_order_device_code_20261013.json)。

旧新版**274条件でLOS/反射点/遅延がbit一致**。
[追加41 tests](../tests/test_bvh_tiny_order.py)は独立stateless GPUをtoleranceなしのoracleに使う。
63/64/65・96・127/128、4,096/4,097・16,384/16,416 rays、単一rayのpositive tie、
極端な全triangle geometry、ECEF/receiver更新/nonfinite、候補並列/serial遷移、返却配列所有、
lazy allocation/容量再利用/解放、deep stack/shared/overlapping leafのfallbackを確認した。
既存3ファイルのメモリ期待値を適用下限64に合わせた。関連**2,204 tests pass**。
追加41 testsのmemcheck/racecheck/synccheckもpass、errors/hazards/leaks 0。
最新packed-satellite PFとsignal batchはsource/build/installedのSHA一致で保持した。
[最終整合確認](bvh_tiny_order_completion_check_20261013.json)。

[32〜63三角形を対象にした3試案](bvh_compact_order_rejected_20261013.json)は不採用。
下限32を無条件適用すると、32/48三角形・16,384 raysの初回処理が10.8%/15.9%増え、全3組で悪化した。
大型candidate下限64を残す案も、48三角形・4,096 raysの初回境界で利得を確認できなかった。
小batchを1,024 raysまでに制限する案はhostの時間変動が大きく、追加条件を採用する根拠が不足した。
3試案はGPU式・10 SASS/resourceを保持し、292/304/312 bit条件と全timed出力が一致。
source/pyd/helper/raw/patchをarchiveし、不採用後のregression/sanitizersは実行していない。

CUDA12.8 / MSVC Release / compute7.5 / C++20 / no fast math。
長時間実データの測位誤差分布やoccupancy/Nsightは今回新しく測定していない。

```powershell
python benchmarks/bench_bvh_tiny_order_compare.py --baseline dist/bvh_tiny_order_baseline/_bvh.cp312-win_amd64.pyd --candidate build/diag/Release/_bvh.cp312-win_amd64.pyd --baseline-dir dist/bvh_tiny_order_baseline --counts 63 64 96 127 128 --shapes 1x1 1x16 129x32 512x32 --rounds 5 --repeats 20 --output dist/bvh_tiny_order_repeat.json
```

## Small resident meshes reuse original GPU normals and ordered candidates (2026-10-13)

直前は参照三角形512未満のresident ownerが、各rayで三角形法線とBVH訪問を再計算していた。
hostの適用下限2箇所を**512→128**へ変更し、128〜511三角形にも既存の順序capture、GPU法線再利用とcandidate評価を適用。
全候補・元の訪問順を保ち、GPU式は変更しない。
511三角形・1epoch×16衛星の公開LOS+反射全体は
**2.2595→0.1373ms（約93.9%短縮）**。
129epoch×32衛星は**6.6267→3.0106ms（約54.6%短縮）**。
対象3メッシュ×4 ray batch×3 stageの36条件で、5交互round×20の全5組が改善した。
1/16 rays、4,128 rays、上限16,384 rays、初回owner生成・trace・closeを含めて確認。

| 条件 | 直前 (ms) | 新 (ms) | 時間短縮 |
|:---|---:|---:|---:|
| 128三角形 / 公開1epoch×16衛星 | 0.6241 | 0.1059 | 83.0% |
| 256三角形 / 公開1epoch×16衛星 | 1.1637 | 0.1089 | 90.6% |
| 511三角形 / 公開1epoch×16衛星 | 2.2595 | 0.1373 | 93.9% |
| 511三角形 / 公開1epoch×1 ray | 2.1692 | 0.0994 | 95.4% |
| 128三角形 / 公開129epoch×32衛星 | 1.8079 | 0.9808 | 45.8% |
| 511三角形 / 公開129epoch×32衛星 | 6.6267 | 3.0106 | 54.6% |
| 511三角形 / 上限512epoch×32衛星 | 19.4068 | 11.1275 | 42.7% |
| 511三角形 / 初回生成・trace・close / 1epoch×16衛星 | 2.3644 | 0.2490 | 89.5% |
| 127三角形 / 未変更公開1epoch×16衛星 | 0.6163 | 0.6159 | 0.1% |
| 512三角形 / 未変更公開129epoch×32衛星 | 3.0010 | 2.9970 | 0.1% |
| 511三角形 / 未変更LOS129epoch×32衛星 | 0.2175 | 0.2178 | -0.1% |

公開warmはBVHAccelerator.trace_batchの入力検査・H2D・LOS/反射全kernel・同期・D2H・返却配列を含む。
native coldは構築済みCPU nodes/trianglesからowner生成・upload・初回準備・trace・closeを含む。
CPU BVH構築と出力hashは時間外。実production旧新pydを同processで順次実行し、call/roundごとに交互にした。
全timed出力をbit hash照合。[80 timed条件・20メモリ条件のraw](bvh_small_order_gpu_20261013.json)。
hostは非占有。小メッシュの改善率を大型都市/PF全体へ一般化せず、過去の速度比と掛け合わせない。

旧**10 GPU関数のSASS instruction bodyとresource countsはすべて同一**。
FP64のnormal/mirror/交差/反射点/正遅延、元のright-first訪問ordinal/tie、sp<62とnonfinite ray zerosを維持。
候補数・ray数・丸め・物理モデルを削減しない。stateless APIの下限512も維持し、独立した旧GPU oracleとして照合した。
orderを安全にcaptureできないshared/overlapping leaf等のfallbackを維持。
128〜511参照三角形にはorder4＋normal32 bytes/visitをlazyに確保する。
4,097〜16,384 raysの追加winnerは**40 bytes/ray、最大640KiB**。
小batchの既存tile数、8,192 visits以上のoptional mirrorと32MiB上限、16,384を超えるcached serialの条件は維持。
LOS-onlyはreflection cacheを作らず、owner内で容量再利用、close/releaseですべて解放する。
[source/pyd/resource/SASS・正確なsource/test再構成patch](bvh_small_order_device_code_20261013.json)。

旧新版**256条件でLOS/反射点/遅延がbit一致**。
[追加40 tests](../tests/test_bvh_small_order.py)は保持したstateless GPUを独立oracleに使い、127/128/129・255/256・511/512、
16/4,128/16,384/16,416 rays、単一rayのpositive tie、極端な全triangle geometry、ECEF/receiver更新/nonfinite、
parallel/serial遷移、返却配列所有、lazy allocation/容量再利用/640KiB上限/解放をtoleranceなしで確認。
既存2メモリ検査は新規対象511のorder/normal/winnerも含む正確な期待値へ更新した。
関連**2,163 tests pass**。追加40 testsのmemcheck/racecheck/synccheckもpass、errors/hazards/leaks 0。
最新packed-satellite PFとsignal batchはsource/build/installedのSHA一致で保持。
[最終整合確認](bvh_small_order_completion_check_20261013.json)。

今回の[PF scan一時領域再利用案](pf_scan_cache_rejected_20261013.json)は不採用。
GPU65 SASS/resourceを維持したが、再サンプリング単体の再確認8.0%/11.7%短縮に対して、
PR+AFV全体は本計測0.24%増、交互call再確認6.25%/4.16%短縮、2回目の未変更controlも6.59%短縮と揺れた。
更新全体の安定した追加利得を十分に識別できず、旧packed source/pydへ正確に戻した。
11,282比較中、既存PR/DD/AFV等10,296比較（入力棄却432を含む）は一致。
Systematicの8 state差は既存のparallel CDF境界変動で、旧新版それぞれの同一build反復でも祖先変動を確認。
全面的なbit一致とは扱わず、両buildのCDF/normalized weights/ancestor診断をrawへ記録した。
全source/pyd/helper/probe/raw2再確認/patchと未実行draft24testsをarchiveし、performance gateの不採用後に
そのdraft regression/sanitizersは走らせていない。

CUDA12.8 / MSVC Release / compute7.5 / C++20 / no fast math。
長時間実データの測位誤差分布やoccupancy/Nsightの新しい測定は行っていない。

```powershell
python benchmarks/bench_bvh_small_order_compare.py --baseline dist/bvh_small_order_baseline/_bvh.cp312-win_amd64.pyd --candidate build/diag/Release/_bvh.cp312-win_amd64.pyd --baseline-dir dist/bvh_small_order_baseline --counts 127 128 256 511 512 --shapes 1x1 1x16 129x32 512x32 --rounds 5 --repeats 20 --output dist/bvh_small_order_repeat.json
```

## Medium resident reflection batches use existing bounded parallelism (2026-10-13)

直前は訪問三角形8,192以上に限っていた4,097〜16,384 raysの並列反射を、
ordered visits **512以上**へ広げた。変更はhost dispatchの1行。
同じ全候補を既存のcandidate/finalize GPU関数へ分配する。
公開LOS+反射全体は512三角形・129epoch×32衛星で
**4.0883→3.0084ms（約26.4%短縮）**、
4,096三角形の同条件で
**30.4804→20.7920ms（約31.8%短縮）**。
対象5メッシュ規模×3 batch×3 stageの45条件を5交互round×20で計測。
公開warm15条件とnative反射15条件、初回14条件の計44条件で全5組が改善。
上限16,384 raysでも改善したが、小メッシュの利得は小さく、
512三角形では11.8848→11.3233ms（約4.7%短縮）。
例外として512三角形・16,384 raysのnative初回生成・trace・closeは
12.0086→12.1063ms（約0.10ms、0.8%増）で全5組が悪化した。
この最小mesh/最大batchでは反復処理の短縮に対して初回準備の負担があるため、coldの高速化は主張しない。
16,384を超えるbatchの元のcached serial、512未満のfallback、stateless APIは維持。

| 条件（32衛星） | 直前 (ms) | 新 (ms) | 時間短縮 |
|:---|---:|---:|---:|
| 512三角形 / 公開129epoch | 4.0883 | 3.0084 | 26.4% |
| 1,200三角形 / 公開129epoch | 9.1972 | 6.5080 | 29.2% |
| 2,048三角形 / 公開257epoch | 30.0642 | 21.1351 | 29.7% |
| 4,096三角形 / 公開129epoch | 30.4804 | 20.7920 | 31.8% |
| 8,191三角形 / 公開257epoch | 119.8492 | 82.2147 | 31.4% |
| 512三角形 / 上限512epoch | 11.8848 | 11.3233 | 4.7% |
| 4,096三角形 / 上限512epoch | 89.5072 | 81.0609 | 9.4% |
| 4,096三角形 / 初回生成・trace・close129epoch | 31.4372 | 21.7184 | 30.9% |
| 512三角形 / 初回生成・trace・close512epoch | 12.0086 | 12.1063 | -0.8% |
| 511三角形 / 未変更の公開129epoch | 6.6270 | 6.6395 | -0.2% |
| 4,096三角形 / 未変更LOS129epoch | 0.4268 | 0.4249 | 0.5% |

公開warmはBVHAccelerator.trace_batchの入力検査・H2D・LOS/反射全kernel・同期・D2H・返却配列まで含む。
native coldは構築済みCPU nodes/trianglesからowner生成・upload・初回準備・trace・closeを含む。
CPU BVH構築とhash照合は時間外。実production旧新pydを同processに読み、GPUは順次、call/roundごとに交互実行。
全timed出力をbit hash照合。[72 timed条件・18メモリ条件のraw](bvh_medium_parallel_gpu_20261013.json)。
4,096→2,048→512 visitsと下限を順に調べた3×5予備測定とsource/pyd/helperのarchive SHAも
[device記録](bvh_medium_parallel_device_code_20261013.json)へ保存した。
hostは非占有。過去の速度比と掛け合わせず、都市シミュレーション/PF全体の速度比へ一般化しない。

旧**10 GPU関数のSASS instruction bodyとresource countsがすべて同一**。device関数の追加はない。
FP64のnormal/mirror/交差/反射点/正遅延、元のright-first訪問ordinalとtie、sp<62捕捉、nonfinite rayのゼロを維持。
候補数・ray数・丸め・物理モデルを減らさない。
新規対象512〜8,191 visitsはmirror準備の下限未満なので、追加はlazyな**40 bytes/ray、最大640KiB**のwinner scratchのみ。
既存normal/order cache、8,192 visits以上のoptional mirrorと32MiB上限は維持。
LOS-onlyはreflection scratchを作らず、owner内で容量を再利用し、close/releaseですべて解放。
[source/pyd/resource/SASS・CRLF再構成patch](bvh_medium_parallel_device_code_20261013.json)。

旧新版**220条件でLOS/反射点/遅延がbit一致**。
[追加45 tests](../tests/test_bvh_medium_parallel.py)は保持したstateless GPUを独立oracleに使い、
511/512/513と4,096/4,097・16,384/16,385の境界、極端なgeometry、tie、ECEF、receiver更新、
parallel/serial遷移、返却配列所有、lazy allocation、640KiB上限と解放をtoleranceなしで確認。
既存2箇所のメモリ検査も、新たなwinner容量を含む正確な期待値に更新した。
関連**2,123 tests pass**。追加45 testsのmemcheck/racecheck/synccheckもpass、errors/hazards/leaks 0。
最新packed satellite PFとsignal batchのsource/build/installedのSHA一致も確認。
[最終整合確認](bvh_medium_parallel_completion_check_20261013.json)。
CUDA12.8 / MSVC Release / compute7.5 / C++20 / no fast math。
長時間実データの測位誤差分布やoccupancy/Nsightの新しい測定は行っていない。

```powershell
python benchmarks/bench_bvh_mesh512_compare.py --baseline dist/bvh_medium_parallel_baseline/_bvh.cp312-win_amd64.pyd --candidate build/diag/Release/_bvh.cp312-win_amd64.pyd --baseline-dir dist/bvh_medium_parallel_baseline --counts 511 512 1200 2048 4096 8191 --shapes 129x32 257x32 512x32 --rounds 5 --repeats 20 --output dist/bvh_medium_parallel_repeat.json
```

## PF PR/GMM/carrier observations use one packed H2D transfer (2026-10-13)

擬似距離PR・GMM・undifferenced carrier AFVは、毎回同じ5列データを3回に分けてGPUへ送っていた。
同じ5*nSat doublesを1つのdevice allocationへ送り、観測値と重みはその内部のaliasで渡す。
owner生成・growth・解放時のsatellite device allocation/freeも3つから1つへ減らした。
演算・データ量・APIは維持。小規模PFの転送/API負担を減らす改善。

2,048粒子・12衛星の公開predict→PR update→AFV update→estimate（各updateのESS/条件付きresampleを含む）は、
起動順反転・callごと交互の再確認で**0.4501→0.4045ms（約10.1%短縮）**。
同じ再確認のPR Gaussian/GMM/MUPFの各全5 roundで改善。
通常のstageごと交互な本計測のMUPF全5 roundでも改善した。

| 条件 | 旧 (ms) | 新 (ms) | 速度比 |
|:---|---:|---:|---:|
| 2,048粒子 / 公開PR Gaussian step再確認 | 0.2533 | 0.2326 | 1.089× |
| 2,048粒子 / 公開GMM step再確認 | 0.3142 | 0.2863 | 1.097× |
| 2,048粒子 / 公開PR+AFV MUPF step再確認 | 0.4501 | 0.4045 | 1.113× |
| 2,048粒子 / 公開PR+AFV MUPF step本計測 | 0.5378 | 0.4526 | 1.188× |
| 10,000粒子 / 公開PR+AFV MUPF step本計測 | 0.7328 | 0.7097 | 1.033× |
| 100万粒子 / 公開PR+AFV MUPF step本計測 | 21.6182 | 21.6544 | 0.998× |
| 100万粒子 / 公開GMM step本計測 | 21.0044 | 21.0787 | 0.996× |

[本計測48条件、4粒子数×12stage、5交互round×20](pf_packed_satellite_gpu_20261013.json)。
実production旧新pydを別の常駐workerで順次実行し、同期wall timeを測る。
warmは入力転送・予測・尤度・ESS・条件付きresample・推定を含み、初期化/reset、IPC、出力hashは時間外。
全warm timed log weightsとreturned estimateをhash照合し、各stage終了時の16成分state/weight/returnも照合。
cold8条件はowner生成・初期化・step・destroy全体を含み、返却ESS/decision/estimateをhash照合する。
coldではfull-state D2H exportを計測せず、state/log weightsのexact照合は別のvalidatorで行う。

本計測にはhost共有に伴う揺れがあるため、
[worker起動順をcandidate firstへ反転し、profile callごと交互にした独立再確認](pf_packed_satellite_recheck_20261013.json)を実施。
2,048粒子6stage、5round×20、各profileで3warmup＋1timed call。初期化/export/destroy/IPCは時間外。
[再確認helper](bench_pf_packed_satellite_recheck.py)も保存。
最初の同方式の再確認もprivate archiveにsource/report/logを保存し、MUPF全体約9.2%短縮を確認した。
最新再確認のPR/GMM/MUPF paired速度比はそれぞれ、
1.054〜1.130 / 1.057〜1.170 / 1.031〜1.143×。
未変更DD/resample対照の差は利得に数えない。

Student公開stepの本計測は0.4689→0.3882msだったが、最新再確認は0.2994→0.3003msで利得を再現できなかった。
10万/100万粒子ではGPU演算が支配し、公開全体はほぼ不変。cold全体にも安定した大きな利得はなく、
allocation/free呼び出し数の減少だけでcold高速化を主張しない。
小規模PF向けの実測として扱い、大規模PF・都市全体・過去の速度比へ一般化しない。

旧**65 GPU関数全てのSASS instruction bodyとresource countsが同一**、新device関数なし。
FP64式、候補/観測/粒子数、RNG、gate/fallback、PR/GMM overwriteとAFV prior additionは保持。
headerはownership commentのみの変更で、declarationとABIは同じ。
同じ5*capacity doublesを1つのdevice slabが所有し、各callのnSatに応じてaliasを3*nSat/4*nSatへ更新。
growth/destroyはbaseだけを解放。pinned sourceの上書き前のstream同期を維持し、DD/Doppler scratchは独立。
要求payload bytesは増やしていない。[source/pyd/resource/SASS/CRLF再構成patch](pf_packed_satellite_device_code_20261013.json)。

旧新**10,296比較条件**（**9,864 bit-output条件＋432 PR入力棄却**）で一致。
既存5epoch軌跡とPR/GMM/DD/予測/再サンプル経路に加え、PR/GMM/AFV切替・衛星数1/4/5/12/33/64/65/128/1,228の履歴を照合。
[追加25 tests](../tests/test_pf_packed_satellite.py)でfresh ownerとgrowth/shrink後のownerを比較し、
alias位置更新・方式切替・queued入力の所有・DD scratchとの独立・empty native GMM/AFVをbit比較。
関連**2,078 tests pass**、25testsのmemcheck/racecheck/synccheckもpass、errors/hazards/leaks0。
最新large-parallel BVHとsignal batchのsource/build/installedはSHA一致で維持。
[最終整合確認](pf_packed_satellite_completion_check_20261013.json)。
CUDA12.8 / MSVC Release / compute7.5 / C++20 / no fast math。
SASS/resourceは静的確認。長時間実データの測位誤差分布は、この変更で再計測していない。

```powershell
python benchmarks/bench_pf_packed_satellite_compare.py --baseline dist/pf_packed_satellite_baseline/_gnss_gpu_pf_device.cp312-win_amd64.pyd --candidate build/diag/Release/_gnss_gpu_pf_device.cp312-win_amd64.pyd --baseline-dir dist/pf_packed_satellite_baseline --sizes 2048 10000 100000 1000000 --rounds 5 --repeats 20 --output dist/pf_packed_satellite_repeat.json
python benchmarks/bench_pf_packed_satellite_recheck.py --baseline dist/pf_packed_satellite_baseline/_gnss_gpu_pf_device.cp312-win_amd64.pyd --candidate build/diag/Release/_gnss_gpu_pf_device.cp312-win_amd64.pyd --baseline-dir dist/pf_packed_satellite_baseline --reference dist/pf_packed_satellite_repeat.json --output dist/pf_packed_satellite_recheck_repeat.json
```

## Larger resident reflection batches use bounded candidate parallelism (2026-10-13)

直前の大型batch用cached serialは1 thread/rayで全候補を評価していた。
訪問三角形8,192以上・total rays4,097〜16,384に限り、既存の1 tile/ray並列candidate評価とwinner確定を使う。
GPUの計算式を変更せず、三角形候補の仕事を多くのblockへ配るhost dispatchの変更。
12,000三角形・129epoch×32衛星の公開LOS+反射全体は
**89.0147→59.9158ms（約32.7%短縮）**。
257epoch×32衛星は**175.6105→120.3161ms（約31.5%短縮）**。
2メッシュ規模×3大型batchの公開/native反射/初回処理の18条件で、5交互round×20の全5組が改善。
主条件のpaired速度比は1.485〜1.486×。
上限16,384 raysの追加3交互round×5でも公開全体は
260.7292→233.6789ms（約10.4%短縮）。
16,416 raysは元のcached serialへ戻り、公開260.4746→260.4486msとほぼ同じ。
[上限付近8 timed条件・2メモリ条件](bvh_large_parallel_upper_probe_20261013.json)と
[独立した再確認helper](bench_bvh_large_parallel_upper_check.py)を保存した。

| 条件 | 旧 (ms) | 新 (ms) | 速度比 |
|:---|---:|---:|---:|
| 8,192三角形 / 公開129epoch×32衛星 | 60.4254 | 36.6630 | 1.648× |
| 12,000三角形 / 公開129epoch×32衛星 | 89.0147 | 59.9158 | 1.486× |
| 12,000三角形 / 公開257epoch×32衛星 | 175.6105 | 120.3161 | 1.460× |
| 12,000三角形 / 公開1epoch×4,097 rays | 87.3302 | 53.3050 | 1.638× |
| 12,000三角形 / native反射129epoch×32衛星 | 88.2111 | 59.1101 | 1.492× |
| 12,000三角形 / native初回生成・trace・close129epoch×32衛星 | 90.6105 | 61.5940 | 1.471× |
| 12,000三角形 / native初回生成・trace・close257epoch×32衛星 | 177.3157 | 123.3194 | 1.438× |
| 1,200三角形 / 未変更の公開129epoch×32衛星 | 9.1550 | 9.1553 | 1.000× |
| 12,000三角形 / 未変更LOS129epoch×32衛星 | 0.8931 | 0.9063 | 0.985× |

公開warmはBVHAccelerator.trace_batchの入力検査・H2D・LOS/反射全kernel・同期・D2H・返却配列を含む。
native coldは構築済みCPU nodes/trianglesからowner生成・upload・初回準備・trace・closeを含む。
CPU BVH構築と出力hashは時間外。実production旧新pydを同processに読み、GPUは順次、call/roundごとに交互実行。
全timed LOS/反射点/遅延をbit hash照合。[48 timed条件・12メモリ条件のraw](bvh_large_parallel_gpu_20261013.json)。
hostは非占有。過去の速度比と掛け合わせず、この条件の実測として扱う。
都市シミュレーション全体やPF全体への速度比はこの比較では計測していない。

旧**10 GPU関数全てのSASS instruction bodyとresource countsが同一**、新device関数は追加していない。
FP64のnormal/mirror/交差/反射点/正遅延、元のright-first訪問ordinalのwinner/tie、sp<62捕捉を保持。
候補の数・ray数・物理モデル・丸めを変えない。対象外のmesh/batchとstateless APIは直前の経路を維持。
[source/pyd/resource/SASS/CRLF再構成patch](bvh_large_parallel_device_code_20261013.json)。

新規対象ではwinner scratchが40 bytes/ray、最大**640KiB**。
元からあるoptional epoch mirror scratchは**32MiB上限**を維持し、各callのreceiverから再構築する。
12,000三角形・129/257epoch×32衛星はmirror容量条件を満たさず、追加はwinner 165,120/328,960 bytes。
8,192三角形・129epoch×32衛星では既存mirror経路も使い、約24.2MiBとwinnerを追加する。
order/normal cacheは従来通り。LOS-onlyはreflection scratchを作らず、owner内で容量再利用、close/releaseで全て解放する。

旧新版**187条件でLOS/反射点/遅延がbit一致**。
8,191/8,192のmesh境界、4,096/4,097と16,384/16,385 ray境界、複数epoch、ECEF、nonfiniteを照合。
[追加24 tests](../tests/test_bvh_large_parallel.py)は保持したstateless GPUを独立oracleに使い、境界・極端なgeometry・
tie順・receiver更新・parallel/serial遷移・返却配列所有・lazy allocation・640KiB上限・解放をtoleranceなしで確認。
関連**2,053 tests pass**。追加24 testsのmemcheck/racecheck/synccheckもpass、errors/hazards/leaks0。
最新PR-gate PFとsignal batchはsource/build/installedのSHA一致で保持。
[最終整合確認](bvh_large_parallel_completion_check_20261013.json)。

今回、以下の4案はexact比較に通ったが、処理全体の安定した改善がなく採用しなかった。
native候補・ソース・pyd・raw・再構成patchを保存し、draft unit testsとsanitizersを走らせる前に棄却した。

| 不採用案 | 比較条件 | 測定結果と理由 |
|:---|---:|:---|
| [Student gate確定後のlog省略](pf_student_skip_rejected_20261013.json) | 10,158（PR入力棄却432を含む） | 100万粒子の公開処理17.1399→17.3224ms。追加分岐が利得にならない |
| [PR尤度係数のblock共有](pf_pr_coefficient_cache_rejected_20261013.json) | 10,497（PR入力棄却432を含む） | Gaussian公開全体は悪化、Studentの小差も安定せず。単体4衛星の揺れは採用根拠にしない |
| [32MiBのepoch mirror分割](bvh_serial_mirror_rejected_20261013.json) | 181 bit条件 | 公開129epoch×32衛星88.3615→170.9949ms。分割による並列度低下が大きい |
| [warp leaderのmirror共有](bvh_warp_mirror_rejected_20261013.json) | 181 bit条件 | 公開87.5728→87.7400ms。処理全体はほぼ不変 |

CUDA12.8 / MSVC Release / compute7.5 / C++20 / no fast math。
SASS/resourceは静的確認。occupancyやNsightの測定、長時間実データの測位誤差分布の再評価は主張しない。

```powershell
python benchmarks/bench_bvh_large_parallel_compare.py --baseline dist/bvh_large_parallel_baseline/_bvh.cp312-win_amd64.pyd --candidate build/diag/Release/_bvh.cp312-win_amd64.pyd --baseline-dir dist/bvh_large_parallel_baseline --counts 1200 8192 12000 --shapes 1x4097 129x32 257x32 1x192 --rounds 5 --repeats 20 --output dist/bvh_large_parallel_repeat.json
```

## Large resident reflection batches reuse ordered GPU normals (2026-10-13)

常駐BVHは4,097 rays以上で旧serial反射へ戻り、全rayで三角形法線と木のstack走査を再計算していた。
新経路はGPUで一度求めた法線と、owner構築時に捕捉した元の訪問順を再利用し、1 thread/rayで同じ候補を評価する。
12,000三角形・129epoch×32衛星（4,128 rays）の公開LOS+反射全体は
**148.6805→88.5833ms（約40.4%短縮）**。
257epoch×32衛星（8,224 rays）は**298.7955→174.3706ms（約41.6%短縮）**。
3メッシュ規模×3大型batchの公開/native反射/初回処理全てで、5交互round×20の全5組が改善。
4,128-ray公開paired速度比は1.677〜1.688×。
通常の192-ray経路とLOS-onlyは元の命令列を維持し、対照の揺れを利得へ数えない。

| 条件 | 旧 (ms) | 新 (ms) | 速度比 |
|:---|---:|---:|---:|
| 512三角形 / 公開129epoch×32衛星 | 6.7462 | 4.1713 | 1.617× |
| 1,200三角形 / 公開129epoch×32衛星 | 15.2214 | 9.2068 | 1.653× |
| 12,000三角形 / 公開129epoch×32衛星 | 148.6805 | 88.5833 | 1.678× |
| 12,000三角形 / 公開257epoch×32衛星 | 298.7955 | 174.3706 | 1.714× |
| 12,000三角形 / native反射129epoch×32衛星 | 147.5434 | 87.5350 | 1.686× |
| 12,000三角形 / native初回生成・trace・close129epoch×32衛星 | 151.2878 | 91.4254 | 1.655× |
| 12,000三角形 / native初回生成・trace・close257epoch×32衛星 | 324.5651 | 177.8446 | 1.825× |
| 12,000三角形 / 公開1epoch×4,097 rays | 147.6510 | 86.6638 | 1.704× |
| 12,000三角形 / 未変更の公開1epoch×192 rays | 3.5887 | 3.5673 | 1.006× |
| 12,000三角形 / 未変更LOS129epoch×32衛星 | 0.9030 | 0.8984 | 1.005× |

公開warmはBVHAccelerator.trace_batchで入力検査・H2D・全kernel・同期・D2H・返却配列を含む。
warmではownerと初回に準備した法線を再利用する。
native coldは構築済みCPU nodes/trianglesからowner生成・upload・初回準備・trace・closeを含む。
CPU BVH構築と出力hashは時間外。実production旧新pydを同processに読み、GPUは順次、call/roundごとに順序を交互にする。
全timed LOS/反射点/遅延をbit hash照合。[48 timed条件・12メモリ条件のraw](bvh_serial_cache_gpu_20261013.json)。
hostは非占有。過去の速度比と掛け合わせず、この大きなバッチ条件の実測として扱う。
都市シミュレーション全体やPF全体への速度比は、この比較では計測していない。

参照triangles>=512で元の訪問順を捕捉できる常駐owner、かつtotal rays>4,096に適用。
元のright-first leaf順とstack条件sp<62で捕捉し、深さ制限でskipされる枝も元と同じ。
候補の数・ray数を減らさず、元のFP64 cross/sqrt/normal3division/strict normal cutoffを同じGPUで計算する。
rayごとの鏡像点、方向正規化、交差式、反射点、距離、strict winner比較を元と同じ式・順序で評価。
等しい正遅延は最初の訪問候補を保持。nonfinite ray paddingのzero値も保つ。
small/unordered/重複leaf等のfallback、stateless API、4,096 rays以下のparallel normal/mirror経路は旧GPU関数のまま。

旧**9関数全てのSASS instruction bodyとresource countsが同一**。
新serial-cache kernelはregister85/stack0/local0/shared0。旧serialはregister90/stack256 bytes。
visit order4 bytesとnormal32 bytes、計**36 bytes×訪問entry**を初回reflectionで用意し、owner内で再利用する。
12,000 entriesなら432,000 bytes追加。LOS-onlyは準備せず、release/closeで全て解放する。
新serialの初回はpartial winner/epoch別鏡像pointを確保しない。parallelからの遷移では既存scratchの容量を保持する。
[source/pyd/resource/SASS/CRLF再構成patch](bvh_serial_cache_device_code_20261013.json)。

旧新版**162条件でLOS/反射点/遅延がbit一致**。normal cutoff前後、underflow/overflow、ECEF、nonfinite、
4,096/4,097境界、複数epoch、tie、grow/shrink、owner snapshotを確認。
[追加31 tests](../tests/test_bvh_serial_cache.py)は命令列を保持したstateless GPU経路を独立oracleに使い、
大型batch・NaN padding・strict cutoff・signed zero・deep stack guard・unordered/overlapping/shared leaves・
serial/parallel切替・入力所有・返却配列所有・lazy cache/allocated_bytes/releaseをtoleranceなしで照合する。
既存serialメモリtestの期待値を、reflection初回に36 bytes/entryが追加される仕様へ更新した。
関連**2,029 tests pass**、追加31 testsのmemcheck/racecheck/synccheckもpass、errors/hazards/leaks0。
最新PR-gate PFとsignal batchのsource/build/installedはSHA一致で保持。
[最終整合確認](bvh_serial_cache_completion_check_20261013.json)。
CUDA12.8 / MSVC Release / compute7.5 / C++20 / no fast math。
SASS/resourceは静的確認で、occupancy/Nsightの実測は主張しない。

```powershell
python benchmarks/bench_bvh_serial_cache_compare.py --baseline dist/bvh_serial_cache_baseline/_bvh.cp312-win_amd64.pyd --candidate build/diag/Release/_bvh.cp312-win_amd64.pyd --baseline-dir dist/bvh_serial_cache_baseline --counts 512 1200 12000 --shapes 1x4097 129x32 257x32 1x192 --rounds 5 --repeats 20 --output dist/bvh_serial_cache_repeat.json
```

## Single-pass undifferenced PR gates (2026-10-13)

通常の擬似距離PRのpositive per-particle gateを、Gaussian/Huber/Studentの専用kernelへ振り分ける。
旧実装はgate countと尤度で距離を2回計算していた。新実装はgeometryを1回計算し、
全観測sumと採用観測sumをそれぞれ元の観測順で累積して、最後に元の不足判定を適用する。
100万粒子・12衛星の公開PF全stepはgaussian **14.7978→12.4090ms（約16.1%短縮）**、huber **16.9270→13.8134ms（約18.4%短縮）**、student **19.6077→17.1278ms（約12.6%短縮）**。
3種類の公開/native全stepとも5交互round×20の全5組で改善。公開paired速度比はgaussian 1.189〜1.197×、huber 1.224〜1.226×、student 1.139〜1.149×。
小粒子数と変更対象外controlの揺れを一律の利得として扱わない。

| 条件 | 旧 (ms) | 新 (ms) | 速度比 |
|:---|---:|---:|---:|
| 1万粒子 / 公開PR gaussian全step | 0.3391 | 0.2962 | 1.145× |
| 1万粒子 / 公開PR huber全step | 0.3740 | 0.3311 | 1.129× |
| 1万粒子 / 公開PR student全step | 0.4530 | 0.4234 | 1.070× |
| 10万粒子 / 公開PR gaussian全step | 1.4385 | 1.1843 | 1.215× |
| 10万粒子 / 公開PR huber全step | 1.8176 | 1.3793 | 1.318× |
| 10万粒子 / 公開PR student全step | 2.0126 | 1.7297 | 1.164× |
| 100万粒子 / 公開PR gaussian全step | 14.7978 | 12.4090 | 1.193× |
| 100万粒子 / 公開PR huber全step | 16.9270 | 13.8134 | 1.225× |
| 100万粒子 / 公開PR student全step | 19.6077 | 17.1278 | 1.145× |
| 100万粒子 / native PR gaussian全step | 14.7994 | 12.3910 | 1.194× |
| 100万粒子 / native PR huber全step | 17.1769 | 14.7761 | 1.162× |
| 100万粒子 / native PR student全step | 19.5947 | 17.1534 | 1.142× |
| 100万粒子 / PR gaussian重み更新12衛星 | 7.3559 | 4.2188 | 1.744× |
| 100万粒子 / PR huber重み更新12衛星 | 8.4786 | 5.3937 | 1.572× |
| 100万粒子 / PR student重み更新12衛星 | 10.9649 | 8.5127 | 1.288× |
| 100万粒子 / 変更対象外の公開GMM全step | 21.1316 | 21.2081 | 0.996× |
| 100万粒子 / 変更対象外のmixed joint DD全step | 17.0143 | 16.9624 | 1.003× |
| 100万粒子 / 変更対象外のStudent ungated | 8.2109 | 12.5710 | 0.653× |
| 100万粒子 / 変更対象外のB60 resample | 3.2206 | 2.1131 | 1.524× |

初回の変更対象外controlはStudent_control 8.2109→12.5710ms、resample_control 3.2206→2.1131msと大きく変動した。
worker作成・交互実行の先頭を反転した追加5×20ではStudent_control 10.0903→9.7063ms、resample_control 4.0429→3.9662msとなり、初回の差を再現しなかった。
両計測の出力はbit一致。同じGPU命令列・resource countsでもこのhostでは時間が揺れるため、
controlの速度差を今回の利得へ含めない。[逆順controlのrawとhelper SHA](pf_pr_gate_control_recheck_20261013.json)。

全stepはpredict/PR weight/ESS判定/条件付きB60 resample/estimateを含む。
sigma5、threshold30m、Huber k1.5、Student nu3。12衛星の3衛星に50mのoutlierを加え、
position spread10m/clock spread30mの粒子に独立した選別・fallbackが生じる条件を使う。
単独重み更新にもpacking/H2D/kernel/同期を含む。初期化・state復元・IPC・hash照合は時間外。
実production旧新pydを別persistent workerへ読み込み、GPUを順次実行してstage/roundごとに順序を交互にする。
全timed log weights・返却値・最終16状態をbit hash照合した。
hostは非占有。[4粒子数・76条件のraw](pf_pr_gate_gpu_20261013.json)。過去の速度比とは掛け合わせない。

positive thresholdのみ適用。nu>0ならStudentを優先し、それ以外は有効Huber、残りはGaussian。
ungated経路/API/既定設定を変更しない。GPUの元のFP64 geometry/時計誤差/weight積/通常logを保ち、
各sumのFMAと観測順を維持する。`<= threshold` でcount、`!(> threshold)` でfiltered sumへ加算し、
元の `min(nSat,4)` に不足すれば全観測sumを選ぶ。PR log weightsの上書きも維持する。
native bindingがnegative weights/nonfinite observationsを拒否する動作も維持する。
dynamic sharedは元の5*nSat doublesのまま、block256/barrier1回。
H2D/allocation/launch数を変更しない。1,228衛星の48KiB上限付近も確認した。

既存**62個全てのSASS instruction bodyとresource countsが同一**。
新Gaussian/Huber/Studentはregister42/51/44、stack/local/static shared0。
[source/pyd/resource/SASS/CRLF再構成patch](pf_pr_gate_device_code_20261013.json)。
旧新**10,158比較条件が一致**（9,726出力bit比較、432PR入力拒否ケースの例外・state/log weights比較）。
追加1,527条件はStudent/Huber優先順位、nu/k/sigma極端値、zero/negative weights、geometry overflow、
nonfinite state/obs、最小4観測の成否、1,228衛星の容量境界・grow/shrink、5epoch軌跡を含む。
実際のpublic classを実装するruntime/config/smootherのSHAもhelper依存として固定した。
[追加109 tests](../tests/test_pf_pr_gate.py)は命令列を保持したungated GPU経路に採用subsetを渡す独立oracle。
inverse sumやtoleranceを使わず、観測順、部分block、上限、zero weights、非有限時計のNaN bits、
複数owner、係数変更・成長・上書き、threshold境界と粒子別fallbackをbit照合する。
関連**1,719 tests pass**、追加109 testsのmemcheck/racecheck/synccheckもpass、errors/hazards/leaks0。

途中のexplicit rounded multiply/subtract案は、元のptxasが融合するFMAを妨げ、85条件でbit不一致となったため却下。
PTXだけでは最終融合を判断できないのでSASSと個別cost/累積出力を確認し、採用版は元のFMAを維持した。
失敗版のsource/pyd/helper/raw/patchを固定し、計測・sanitizerには進めていない。
[却下記録](pf_pr_gate_rounding_rejected_20261013.json)。
前回GMM/DDの命令列を保持し、BVH native mirror-sharingとsignal batchのsource/build/installed SHAも維持。
[最終整合確認](pf_pr_gate_completion_check_20261013.json)。
CUDA 12.8 / MSVC Release / compute 7.5 / C++20 / no fast math。
SASS/resourceは静的確認で、occupancy実測やNsight結果は主張しない。

```powershell
python benchmarks/bench_pf_pr_gate_compare.py --baseline dist/pf_pr_gate_baseline/_gnss_gpu_pf_device.cp312-win_amd64.pyd --candidate build/diag/Release/_gnss_gpu_pf_device.cp312-win_amd64.pyd --baseline-dir dist/pf_pr_gate_baseline --sizes 2048 10000 100000 1000000 --rounds 5 --repeats 20 --output dist/pf_pr_gate_repeat.json
```

## Exact GMM coefficients and finite-logit single-exp (2026-10-13)

GMMの全粒子共通係数（GPUのsigma reciprocal2個・log4個）をblockで1回計算し、既存barrierで共有する。
さらに両logitが有限な場合、最大成分の `exp(max-max)` は元から厳密に1なので、残る成分のexpだけを求める。
100万粒子・12衛星の公開PF全stepは**24.8145→21.0389ms（約15.2%短縮）**。
公開/nativeとも5交互round×20の全5組で改善。公開paired速度比は1.177〜1.259×。
両成分を同じsigma/meanにした全stepも24.8494→21.0947msで全5組改善。
小粒子数と変更対象外controlの揺れを一律の利得として扱わない。

| 条件 | 旧 (ms) | 新 (ms) | 速度比 |
|:---|---:|---:|---:|
| 1万粒子 / 公開GMM全step | 0.4492 | 0.3776 | 1.189× |
| 10万粒子 / 公開GMM全step | 2.5220 | 2.1558 | 1.170× |
| 100万粒子 / 公開GMM全step | 24.8145 | 21.0389 | 1.179× |
| 100万粒子 / native GMM全step | 24.6965 | 20.9676 | 1.178× |
| 100万粒子 / 公開equal-component GMM全step | 24.8494 | 21.0947 | 1.178× |
| 100万粒子 / GMM重み更新4衛星 | 10.4810 | 6.4276 | 1.631× |
| 100万粒子 / GMM重み更新12衛星 | 16.1369 | 12.3858 | 1.303× |
| 100万粒子 / GMM重み更新32衛星 | 39.8088 | 32.3440 | 1.231× |
| 100万粒子 / GMM重み更新65衛星 | 78.9436 | 65.2995 | 1.209× |
| 100万粒子 / 変更対象外のmixed joint Gaussian全step | 16.8792 | 16.9165 | 0.998× |
| 100万粒子 / 変更対象外のmixed joint Huber全step | 20.4383 | 20.4065 | 1.002× |
| 100万粒子 / 変更対象外のStudent | 11.6986 | 13.2561 | 0.883× |
| 100万粒子 / 変更対象外のB60 resample | 2.7470 | 4.6390 | 0.592× |

全stepはpredict/GMM weight/ESS判定/条件付きB60 resample/estimateを含む。
通常はLOS sigma5、w_LOS0.7、NLOS mean15/sigma30。equal条件は両sigma5、w_LOS0.5、NLOS mean0。
単独重み更新にもpacking/H2D/kernel/同期を含む。初期化・state復元・IPC・hash照合は時間外。
実production旧新pydを別persistent workerへ読み込み、GPUを順次実行してstage/roundごとに順序を交互にする。
全timed log weights・返却値・最終16状態をbit hash照合した。
hostは非占有。[4粒子数・44条件のraw](pf_gmm_gpu_20261013.json)。過去の速度比とは掛け合わせない。

1〜1,227衛星、normal正・有限の両sigma、0<w_LOS<1、有限NLOS meanに適用。
その他のmetadataと1,228衛星のshared-memory上限付近は元のkernelを使う。
係数は元と同じFP64式をGPUで計算し、LOS/NLOS logit・距離・時計誤差・weight積・観測順を維持。
finite logitのexp入力は元の小さい成分 minus maxと同一、元のもう一方はexp(0)=1。
正のexp値のFP64加算を保ち、通常の `log` を使う。`log1p` や近似式へは置き換えない。
非有限logitは元の2exp式とoperand順へ戻し、Inf/NaN/zero-weight overflowのIEEE動作を維持する。
元のordered sumでlog weightsを上書きする。
dynamic sharedは元の5*nSat doublesへ6 doubles（48byte）追加、block256/barrier1回。
H2D/allocation/launch数/API/既定設定を変更しない。1,227衛星も48KiB内。

既存**61個全てのSASS instruction bodyとresource countsが同一**。
新GMMはregister48、stack/local/static shared0。
[source/pyd/resource/SASS/CRLF再構成patch](pf_gmm_device_code_20261013.json)。
旧新**8,631ケースでstate/log weights/全stepがbit一致**。
GMM追加1,533ケースはsigma極端値、mixture0/1/境界/無効値、negative/zero weights、geometry overflow、
nonfinite state/obs、1,227/1,228衛星の容量境界・grow/shrinkを含む。
実際にpublic classを実装するruntime/config/smootherのSHAもhelper依存として固定した。
[追加82 tests](../tests/test_pf_gmm_exact.py)は元のkernelを選ぶ1,228衛星とneutral paddingを独立oracleに使う。
mixture0.5・両sigma1ならpadding costは厳密に0、両sigma2なら元のmixture1 endpointから求めた-log(2)。
このpaddingを同じ順序のFP64加算で付加して元のGPU結果とbit比較する。
inverse sumやtoleranceを使わず、equal/unequal logits、部分block、上限、NaNのbits、係数変更・成長・所有を確認。
通常のsigma5/30・mixture0.7も旧新binaryの厳密照合と全timed出力で確認した。
関連**1,610 tests pass**、追加82 testsのmemcheck/racecheck/synccheckもpass、errors/hazards/leaks0。
前回DD gatesの命令列を保持し、BVH native mirror-sharingとsignal batchのsource/build/installed SHAも維持。
[最終整合確認](pf_gmm_completion_check_20261013.json)。
CUDA 12.8 / MSVC Release / compute 7.5 / C++20 / no fast math。
SASS/resourceは静的確認で、occupancy実測やNsight結果は主張しない。

```powershell
python benchmarks/bench_pf_gmm_compare.py --baseline dist/pf_gmm_baseline/_gnss_gpu_pf_device.cp312-win_amd64.pyd --candidate build/diag/Release/_gnss_gpu_pf_device.cp312-win_amd64.pyd --baseline-dir dist/pf_gmm_baseline --sizes 2048 10000 100000 1000000 --rounds 5 --repeats 20 --output dist/pf_gmm_repeat.json
```

## Single-pass mixed-reference joint DD gates (2026-10-13)

混在参照のPR+搬送波一括更新でも、gateのkept数判定と尤度計算を1回のgeometry loopへ統合した。
元のper-DD参照距離を維持し、gate用と尤度用の重複する距離計算を除く。
100万粒子・12組DDの公開PF全stepはGaussianで**22.0870→16.7508ms（約24.2%短縮）**、
Huberで**25.8809→20.1617ms（約22.1%短縮）**。
両方の公開/native全stepが5交互round×20の全5組で改善した。
公開paired速度比はGaussian 1.160〜1.320×、
Huber 1.281〜1.288×。
小粒子数や変更対象外の揺れを一律の利得として扱わない。

| 条件 | 旧 (ms) | 新 (ms) | 速度比 |
|:---|---:|---:|---:|
| 1万粒子 / 公開mixed joint Gaussian全step | 0.7889 | 0.6876 | 1.147× |
| 10万粒子 / 公開mixed joint Gaussian全step | 3.3644 | 2.4484 | 1.374× |
| 100万粒子 / 公開mixed joint Gaussian全step | 22.0870 | 16.7508 | 1.319× |
| 100万粒子 / native mixed joint Gaussian全step | 22.0384 | 16.6764 | 1.322× |
| 10万粒子 / 公開mixed joint Huber全step | 3.3307 | 2.5175 | 1.323× |
| 100万粒子 / 公開mixed joint Huber全step | 25.8809 | 20.1617 | 1.284× |
| 100万粒子 / native mixed joint Huber全step | 25.9208 | 20.2584 | 1.280× |
| 100万粒子 / mixed joint Gaussian重み更新 | 26.2035 | 17.5848 | 1.490× |
| 100万粒子 / mixed joint Huber重み更新 | 17.2278 | 11.4950 | 1.499× |
| 100万粒子 / 変更対象外のcommon joint Gaussian全step | 14.7004 | 14.7469 | 0.997× |
| 100万粒子 / 変更対象外のcommon joint Huber全step | 18.3703 | 18.2923 | 1.004× |
| 100万粒子 / 変更対象外のmixed個別Gaussian全step | 21.6366 | 21.5746 | 1.003× |
| 100万粒子 / 変更対象外のmixed個別Huber全step | 24.5374 | 24.5479 | 1.000× |
| 100万粒子 / 変更対象外のゲートなし個別全step | 17.0650 | 17.0110 | 1.003× |
| 100万粒子 / 変更対象外のStudent | 10.8624 | 10.3822 | 1.046× |
| 100万粒子 / 変更対象外のB60 resample | 2.6240 | 4.0876 | 0.642× |

全stepはpredict/DD joint重み更新/ESS判定/条件付きB60 resample/estimateを含む。
PR sigmaは公開5/native0.75、CP sigma0.05、PR gate10m/CP gate0.25cycle、Huber kは両方1.5。
参照座標は交互のDDで[12,-4,8]mずらし、対応するbase range/観測も再構成した。
重み更新にもpacking/H2D/kernel/同期を含む。初期化・state復元・IPC・hash照合は時間外。
別persistent workerでGPUを順次実行し、全timed log weights・返却値・最終16状態をbit hash照合。
hostは非占有。[4粒子数・52条件のraw](pf_dd_mixed_joint_gate_gpu_20261013.json)。過去の速度比とは掛け合わせない。

混在参照の2組以上DD・少なくとも片側が正gate・有限packed観測・normal正波長を対象とする。
Huber時はPR/CP両方の有限正kを要求し、片側Huber/k無効は従来のkernelを使う。
single-DD/gateなし/特殊入力、common-reference joint・前回の個別mixed gatesも従来のkernelを維持。
各DDのFP64距離・残差・AFV/rint・観測順、private GPU sigma reciprocalを保つ。
PRは元のweight/cost/sigmaのFMA、CPは元の乗算後の丸めとaccumulator加算を維持する。
Gaussian CPの積は `-0.5*w*afv*afv` の結合順、Huber CPはweight-cost乗算後に丸める。
PRとCPのkept数とfiltered sumは独立し、各sideで `min(nDD,3)` 未満なら全観測へfallback。
元のjointの `<=` filterに従い、成功したgateではNaNをfiltered sumから除外する。
不足時fallbackはNaNを含む元の全観測和を保つ。ゲートなしsideは全観測を使う。
最後のPR+CP加算でlog weightsを上書きする。
dynamic sharedは元の12*nDD doubles、512DDも追加sigma slotなしで48KiBに収まる。
block256/barrier1、H2D/allocation/launch数/API/既定設定を変更しない。

既存**59個全てのSASS instruction bodyが同一**。新Gaussianはregister50、新Huberはregister54、stack/local/static shared0。
[source/pyd/resource/SASS/CRLF再構成patch](pf_dd_mixed_joint_gate_device_code_20261013.json)。
旧新**7,098ケースでstate/log weights/軌跡がbit一致**。
[追加72 tests](../tests/test_pf_dd_mixed_joint_gate.py)は部分block/2〜512DD/独立したPR・CP採用subset/
片側gate/全kept/kept不足/partial NaN/参照・波長・sigma・Huber k変更/成長/上書きを確認する。
元のゲートなしjoint GPU kernelへ各sideの採用subsetを渡し、通常距離のテスト入力で他sideのsigma=1e200によりcostを0にして
元のPR/CP ordered sumを独立に求める。最後に同じFP64加算でbit比較する。
個別APIはCPの丸め方が異なるため、このjoint固有の独立oracleを使った。
従来のcommon対mixed比較も保ち、独立kernelが残るゲートなし比較と旧新binary比較を併用する。
関連**1,528 tests pass**、追加72 testsはmemcheck/racecheck/synccheckでもpass、errors/hazards/leaks0。
BVH native mirror-sharingとsignal batchのsource/build/installed SHAを保持。
[最終整合確認](pf_dd_mixed_joint_gate_completion_check_20261013.json)。
CUDA 12.8 / MSVC Release / compute 7.5 / C++20 / no fast math。
SASS/resourceは静的確認で、occupancy実測やNsight結果は主張しない。

```powershell
python benchmarks/bench_pf_dd_mixed_joint_gate_compare.py --baseline dist/pf_dd_mixed_joint_gate_baseline/_gnss_gpu_pf_device.cp312-win_amd64.pyd --candidate build/diag/Release/_gnss_gpu_pf_device.cp312-win_amd64.pyd --baseline-dir dist/pf_dd_mixed_joint_gate_baseline --sizes 2048 10000 100000 1000000 --rounds 5 --repeats 20 --output dist/pf_dd_mixed_joint_gate_repeat.json
```

## Single-pass mixed-reference DD gates (2026-10-12〜13)

参照衛星座標が混在する個別PR/搬送波更新でも、ゲートのkept数判定と尤度計算を1回のgeometry loopへ統合した。
参照距離は元のとおり各DDで計算し、2回目の同じ距離計算を除く。
100万粒子・12組DDの公開PF全stepはGaussianで**31.4688→21.5671ms（約31.5%短縮）**、
Huberで**34.8032→24.6530ms（約29.2%短縮）**。
両方の公開/native全stepが5交互round×20の全5組で改善した。
公開paired速度比はGaussian 1.455〜1.463×、
Huber 1.410〜1.413×。
小粒子数や変更対象外の揺れを一律の利得として扱わない。

| 条件 | 旧 (ms) | 新 (ms) | 速度比 |
|:---|---:|---:|---:|
| 1万粒子 / 公開mixed Gaussian全step | 0.7412 | 0.5443 | 1.362× |
| 10万粒子 / 公開mixed Gaussian全step | 3.5441 | 2.3308 | 1.521× |
| 100万粒子 / 公開mixed Gaussian全step | 31.4688 | 21.5671 | 1.459× |
| 100万粒子 / native mixed Gaussian全step | 31.4973 | 21.5438 | 1.462× |
| 10万粒子 / 公開mixed Huber全step | 3.5316 | 2.4585 | 1.437× |
| 100万粒子 / 公開mixed Huber全step | 34.8032 | 24.6530 | 1.412× |
| 100万粒子 / native mixed Huber全step | 34.7826 | 24.6385 | 1.412× |
| 100万粒子 / mixed Gaussian PR重み更新 | 11.0745 | 6.2555 | 1.770× |
| 100万粒子 / mixed Gaussian CP重み更新 | 11.7650 | 6.5948 | 1.784× |
| 100万粒子 / mixed Huber PR重み更新 | 13.0003 | 8.7725 | 1.482× |
| 100万粒子 / mixed Huber CP重み更新 | 13.8585 | 9.1010 | 1.523× |
| 100万粒子 / 変更対象外のcommon Gaussian全step | 17.6517 | 17.6177 | 1.002× |
| 100万粒子 / 変更対象外のcommon Huber全step | 20.7362 | 20.7467 | 0.999× |
| 100万粒子 / 変更対象外のjoint Huber全step | 18.4031 | 18.4598 | 0.997× |
| 100万粒子 / 変更対象外のStudent | 14.3042 | 18.8316 | 0.760× |
| 100万粒子 / 変更対象外のB60 resample | 4.0415 | 4.0968 | 0.986× |

全stepはpredict/DD PR（公開sigma5、native sigma0.75、resampleなし）/DD CP（sigma0.05）/
ESS判定/条件付きB60 resample/estimateを含む。PR gate10m/CP gate0.25cycle、Huber kは両方1.5。
参照座標は交互のDDで[12,-4,8]mずらし、対応するbase range/観測も毎回再構成した。
重み更新にもpacking/H2D/kernel/同期を含む。初期化・state復元・IPC・hash照合は時間外。
別persistent workerでGPUを順次実行し、全timed log weights・返却値・最終16状態をbit hash照合。
hostは非占有。10月12日開始のphase IDをraw/patch/logのファイル名に保持した。[4粒子数・52条件のraw](pf_dd_mixed_gate_gpu_20261012.json)。過去の速度比とは掛け合わせない。

混在参照の2組以上DD・正gate・有限packed観測を対象とし、Huber時は有限正k、CPはnormal正波長を要求。
single-DD/gateなし/特殊入力/k無効、既存common-reference/joint更新は従来のkernelを維持した。
元のper-pair FP64距離・残差・AFV/rint・観測順を保つ。
Gaussianのsigma reciprocalは元のGPU式を各粒子で計算し、PRは `w*r*r`、CPは `w*(afv*afv)` を維持。
Gaussianのsigma乗算とsum、Huberのweight-cost sumは元のFMAで計算する。
kept数が `min(nDD,3)` 未満なら全観測のordered sumへfallback。
個別APIのskip-if-greaterに従い、NaNはkept数に含めずfiltered sumには残す。
PRの上書き・CPのprior加算も維持する。
dynamic sharedは元のPR10*nDD/CP11*nDD doubles、block256、barrier1回。
H2D/allocation/launch数/API/既定設定を変更しない。

既存**55個全てのSASS instruction bodyが同一**。新4 kernelsのregister数はGaussian PR44/CP46、Huber PR/CP54。
いずれもstack/local/static shared0。[source/pyd/resource/SASS/CRLF再構成patch](pf_dd_mixed_gate_device_code_20261012.json)。
旧新**6,678ケースでstate/log weights/軌跡がbit一致**。
[追加60 tests](../tests/test_pf_dd_mixed_gate.py)は部分block/2〜512DD/全kept/kept不足/partial NaN/
参照・波長・sigma・Huber k変更/成長/prior加算・上書きを確認する。
成功したpartial gateは、元のゲートなしGPU kernelへ同じ観測順の採用subsetだけを渡してbit比較する。
全kept/kept不足も元のゲートなしGPU costと比較し、common対mixedの照合だけに依存しない。
関連**1,456 tests pass**、追加60 testsはmemcheck/racecheck/synccheckでもpass、errors/hazards/leaks0。
BVH native mirror-sharingとsignal batchのsource/build/installed SHAを保持。
[最終整合確認](pf_dd_mixed_gate_completion_check_20261012.json)。
CUDA 12.8 / MSVC Release / compute 7.5 / C++20 / no fast math。
SASS/resourceは静的確認で、occupancy実測やNsight結果は主張しない。

### Rejected signal batch descriptor cache

同じ検討でbatchの32channel以下・4,096sample以上にdescriptor全体のshared cacheを試した。
旧新633 IQ/metadataケースでbit一致し、既存8 device instruction bodiesを維持した。
新kernelはregister48/static shared2,816byte/stack・local0。
予備3×5ではnative12channelに改善が見えたが、5交互round×20の本測定では安定した処理全体の改善がなかった。
公開8epoch/260,000sample/12channelは19.6815→20.3276ms、都市全体は28.4727→28.4673ms。
変更対象外controlにも揺れがあり、32channelの小さな中央値改善だけで採用しない。
試作を棄却し、元のsource/build/installed binaryをSHA一致で復元した。
[17条件の全raw](signal_batch_descriptor_rejected_20261012.json)、[初期probe](signal_batch_descriptor_probe_20261012.json)、
[device/source/pyd/再構成patch](signal_batch_descriptor_device_code_20261012.json)、[比較runner](bench_signal_batch_descriptor_compare.py)。
棄却版でのCUDA sanitizersは未実施。個別helper/source/pyd/testをdistへ保存し、後続のPF測定とは分離した。

```powershell
python benchmarks/bench_pf_dd_mixed_gate_compare.py --baseline dist/pf_dd_mixed_gate_baseline/_gnss_gpu_pf_device.cp312-win_amd64.pyd --candidate build/diag/Release/_gnss_gpu_pf_device.cp312-win_amd64.pyd --baseline-dir dist/pf_dd_mixed_gate_baseline --sizes 2048 10000 100000 1000000 --rounds 5 --repeats 20 --output dist/pf_dd_mixed_gate_repeat.json
```

## Single-pass individual Huber DD gates (2026-10-12)

PR・搬送波を別々のAPIで更新するHuber gate経路も、kept数判定と尤度計算を1回のgeometry loopへ統合した。
100万粒子・12組DDの公開PF全stepは**26.1994→20.5405ms（1.275×、約21.6%短縮）**。
5交互round×20の全5組で改善し、paired速度比は**1.274〜1.276×**。
native全stepも26.3243→20.6195ms、全5組で改善した。
粒子数ごとの実測値を以下に示す。小粒子数や変更対象外の揺れは一律の利得として扱わない。

| 条件 | 旧 (ms) | 新 (ms) | 速度比 |
|:---|---:|---:|---:|
| 2,048粒子 / 公開Huber PR→CP全step | 0.3235 | 0.3085 | 1.048× |
| 1万粒子 / 公開Huber PR→CP全step | 0.5264 | 0.4268 | 1.233× |
| 10万粒子 / 公開Huber PR→CP全step | 3.0067 | 2.0895 | 1.439× |
| 100万粒子 / 公開Huber PR→CP全step | 26.1994 | 20.5405 | 1.275× |
| 100万粒子 / native Huber PR→CP全step | 26.3243 | 20.6195 | 1.277× |
| 100万粒子 / native Huber PR gate重み更新 | 9.9732 | 6.3106 | 1.580× |
| 100万粒子 / native Huber CP gate重み更新 | 9.1059 | 6.1175 | 1.488× |
| 100万粒子 / 変更対象外のmixed Huber PR→CP全step | 34.7553 | 34.7515 | 1.000× |
| 100万粒子 / 変更対象外のGaussian PR→CP全step | 17.5601 | 17.5448 | 1.001× |
| 100万粒子 / 変更対象外のHuber joint gate全step | 18.3007 | 18.3168 | 0.999× |
| 100万粒子 / 変更対象外のStudent | 8.1665 | 8.1707 | 0.999× |
| 100万粒子 / 変更対象外のB60 resample | 3.2092 | 3.2311 | 0.993× |

公開全stepはpredict/DD PR（sigma5、resampleなし）/DD CP（sigma0.05）/
ESS判定/条件付きB60 resample/estimateを含む。native全stepのPR sigmaは0.75。
Huber kは両方1.5、gate閾値はPR10m/CP0.25cycle。
native重み更新にもpacking/H2D/kernel/同期を含む。
GPUは別persistent workerで順次実行し、全timed log weights・返却値・最終16状態をbit hash照合。
初期化・state復元・IPC・hash照合は時間外、hostは非占有。
変更対象外のmixed/Gaussian/joint/Student/resampleをcontrolとして同時に記録。
[4粒子数・36条件のraw](pf_dd_single_huber_gate_gpu_20261012.json)。過去の速度比とは掛け合わせない。

共通の有限参照座標がbit一致する2組以上DD、有限正Huber k、正gate、有限packed観測だけに適用。
CPにはnormal正波長も要求し、特殊入力/mixed/gateなし/k無効は元のkernelを使う。
元のresidual/sigmaによるHuber cost、weight-costのFMA、観測順を保持する。
全観測/filteredのordered sumとkept数を持ち、`min(nDD,3)` 未満なら元の全観測fallbackを選ぶ。
PRはlog weightsを上書き、CPは既存のlog weightsへ加算する。
個別APIのskip-if-greater条件を維持し、NaNはkept数に入らず、filtered sumには残る。
これはjoint APIのfilter条件とは異なるため、個別APIの元のGPU式とbit一致で確認した。

dynamic sharedはPR10*nDD/CP11*nDD doubles（Huberではsigma cache不要）、block256、barrier1回。
allocation/H2D/launch数/API/既定設定を変更せず、既存の個別Huber gate設定で自動適用。
新PRはregister52、新CPはregister53、いずれもstack/local/static shared0。
既存**53個全てのSASS instruction bodyが同一**で、新しい個別Huber gate kernelを2個追加した。
個別Gaussian・joint Gaussian/Huber、BVH本体とsignal batchのsource/build/installed SHAも維持。
[source/pyd/resource/SASS/CRLF再構成patch](pf_dd_single_huber_gate_device_code_20261012.json)。

旧新**6,486ケースでstate/log weights/軌跡がbit一致**。
[追加62 tests](../tests/test_pf_dd_single_huber_gate.py)は部分block/512DD/kept不足/partial NaN/
Huber k変更とfallback/priorの上書き・加算/参照・波長・sigma変更/バッファ成長を元のGPU式と比較する。
関連**1,396 tests pass**。追加62 testsはmemcheck/racecheck/synccheckでもpass、errors/hazards/leaks0。
[最終整合確認](pf_dd_single_huber_gate_completion_check_20261012.json)。
CUDA 12.8 / MSVC Release / compute 7.5 / C++20 / no fast math。
SASS/resourceは静的確認で、occupancy実測やNsight結果は主張しない。

```powershell
python benchmarks/bench_pf_dd_single_huber_gate_compare.py --baseline dist/pf_dd_single_huber_gate_baseline/_gnss_gpu_pf_device.cp312-win_amd64.pyd --candidate build/diag/Release/_gnss_gpu_pf_device.cp312-win_amd64.pyd --baseline-dir dist/pf_dd_single_huber_gate_baseline --sizes 2048 10000 100000 1000000 --stages PR_huber_gate CP_huber_gate public_split_huber_gate_step native_split_huber_gate_step public_split_huber_gate_mixed_step public_split_gate_step public_joint_huber_gate_step Student_control resample_control --rounds 5 --repeats 20 --output dist/pf_dd_single_huber_gate_repeat.json
```

## Single-pass individual Gaussian DD gates (2026-10-12)

PR・搬送波を別々のAPIで更新するGaussian gate経路も、kept数判定と尤度計算を1回のgeometry loopへ統合した。
100万粒子・12組DDの公開PF全stepは**23.1002→17.4678ms（1.322×、約24.4%短縮）**。
5交互round×20すべてで改善し、paired速度比は**1.319〜1.326×**。
native全stepも23.0912→17.4584ms、全5組で改善した。
10万粒子の公開全stepも全5組で改善したが、小粒子数では改善しないroundがあるため一律の利得を主張しない。

| 条件 | 旧 (ms) | 新 (ms) | 速度比 |
|:---|---:|---:|---:|
| 10万粒子 / 公開PR→CP gate全step | 2.7759 | 1.9842 | 1.399× |
| 100万粒子 / 公開PR→CP gate全step | 23.1002 | 17.4678 | 1.322× |
| 100万粒子 / native PR→CP gate全step | 23.0912 | 17.4584 | 1.323× |
| 100万粒子 / native PR gate重み更新 | 8.0014 | 5.1647 | 1.549× |
| 100万粒子 / native CP gate重み更新 | 7.9272 | 4.6259 | 1.714× |
| 100万粒子 / 変更対象外のmixed PR→CP gate全step | 31.6350 | 31.6163 | 1.001× |
| 100万粒子 / 変更対象外のgateなしPR→CP全step | 17.0300 | 17.0303 | 1.000× |
| 100万粒子 / 変更対象外のHuber joint gate全step | 18.3361 | 18.3141 | 1.001× |
| 100万粒子 / 変更対象外のStudent | 10.7256 | 12.4718 | 0.860× |
| 100万粒子 / 変更対象外のB60 resample | 3.3396 | 3.5989 | 0.928× |

公開全stepはpredict/DD PR（sigma5、resampleなし）/DD CP（sigma0.05）/
ESS判定/条件付きB60 resample/estimateを含む。native全stepのPR sigmaは0.75。
両方のgate閾値はPR10m/CP0.25cycle。native重み更新にもpacking/H2D/kernel/同期を含む。
GPUは別persistent workerで順次実行し、全timed log weights・返却値・最終16状態をbit hash照合。
初期化・state復元・IPC・hash照合は時間外、hostは非占有。
Student/resample controlでも揺れがあり、変更対象外の差は高速化の効果として扱わない。
[4粒子数・36条件のraw](pf_dd_single_gate_gpu_20261012.json)。過去の速度比とは掛け合わせない。

共通の有限参照座標がbit一致する2組以上DD、Huber disabled、正のgate、有限packed観測だけに適用。
CPにはnormal正波長も要求し、特殊入力/mixed/gateなし/Huberは元のkernelを使う。
全観測/filteredのordered sumとkept数を持ち、`min(nDD,3)` 未満なら元の全観測fallbackを選ぶ。
PRは `w*r*r`、CPは `w*(afv*afv)` の元の積の結合順を保ち、sigma乗算とsumには元のFMAを使う。
PRはlog weightsを上書き、CPは既存のlog weightsへ加算する。
個別APIの元のskip-if-greater条件を維持し、NaNはkept数に入らず、filtered sumには残る。
このAPI固有のNaN動作も元のmixed-reference kernelとbit一致でテストした。

dynamic sharedは従来同様PR10*nDD+1/CP11*nDD+1 doubles、block256、barrier1回。
allocation/H2D/launch数/API/既定設定を変更せず、既存Gaussian gate設定で自動適用。
新PRはregister46、新CPはregister40、いずれもstack/local/static shared0。
既存**51個全てのSASS instruction bodyが同一**で、新しい個別gate kernelを2個追加した。
joint Gaussian/Huberの命令列、BVH本体とsignal batchのsource/build/installed SHAも維持。
[source/pyd/resource/SASS/CRLF再構成patch](pf_dd_single_gate_device_code_20261012.json)。

旧新**6,102ケースでstate/log weights/軌跡がbit一致**。
[追加54 tests](../tests/test_pf_dd_single_gate.py)は部分block/512DD/kept不足/partial NaN/
priorの上書き・加算/参照・波長・sigma変更/バッファ成長を元のGPU式と比較する。
関連**1,334 tests pass**。追加54 testsはmemcheck/racecheck/synccheckでもpass、errors/hazards/leaks0。
[最終整合確認](pf_dd_single_gate_completion_check_20261012.json)。
CUDA 12.8 / MSVC Release / compute 7.5 / C++20 / no fast math。
SASS/resourceは静的確認で、occupancy実測やNsight結果は主張しない。

```powershell
python benchmarks/bench_pf_dd_single_gate_compare.py --baseline dist/pf_dd_single_gate_baseline/_gnss_gpu_pf_device.cp312-win_amd64.pyd --candidate build/diag/Release/_gnss_gpu_pf_device.cp312-win_amd64.pyd --baseline-dir dist/pf_dd_single_gate_baseline --sizes 2048 10000 100000 1000000 --stages PR_gate CP_gate public_split_gate_step native_split_gate_step public_split_gate_mixed_step public_CP_step public_joint_huber_gate_step Student_control resample_control --rounds 5 --repeats 20 --output dist/pf_dd_single_gate_repeat.json
```

## Single-pass both-Huber DD gates and partial-NaN correction (2026-10-12)

共通参照DDのPR・搬送波の両方にHuberを使うゲート付き更新でも、geometryを1 loopへ統合した。
100万粒子・12組DDの公開PF全stepは**21.7786→18.2760ms（1.192×、約16.1%短縮）**。
5交互round×20すべてで改善し、paired速度比は**1.188〜1.194×**。
10万粒子も2.1464→1.8077ms（約15.8%短縮）、全5組で改善した。
2,048/10,000粒子はCPU側の揺れが大きく、全roundでの改善は確認できないため一律の利得を主張しない。

| 条件 | 旧 (ms) | 新 (ms) | 速度比 |
|:---|---:|---:|---:|
| 10万粒子 / 公開Huber gate全step | 2.1464 | 1.8077 | 1.187× |
| 100万粒子 / 公開Huber gate全step | 21.7786 | 18.2760 | 1.192× |
| 100万粒子 / native Huber gate全step | 21.7595 | 18.2081 | 1.195× |
| 100万粒子 / native Huber gate重み更新 | 12.9039 | 9.4289 | 1.369× |
| 100万粒子 / Gaussian gate全step | 14.9493 | 14.6692 | 1.019× |
| 100万粒子 / 変更対象外のHuber gate mixed全step | 26.0140 | 25.9802 | 1.001× |
| 100万粒子 / 変更対象外のHuber重み更新（gateなし） | 9.7570 | 11.4273 | 0.854× |
| 100万粒子 / 変更対象外のStudent | 8.0860 | 8.0828 | 1.000× |
| 100万粒子 / 変更対象外のB60 resample | 3.3131 | 3.5726 | 0.927× |

全stepはpredict/DD joint/ESS判定/条件付きB60 resample/estimateを含む。
native重み更新もpacking/H2D/kernel/stream同期まで計測する。
実production pydを別persistent workerに読み込み、GPUを順次実行して旧新順を交互にした。
全timed log weights・返却値と各stage最終16状態をbit hashで照合。
初期化・state復元・IPC・hash照合は時間外、hostは非占有。
gateなしHuberやresampleの変更対象外controlにも揺れがあり、その差は新処理の効果として扱わない。
[4粒子数・32条件・全raw](pf_dd_huber_gate_gpu_20261012.json)。過去の速度比とは掛け合わせない。

### 精度と適用条件

前回のGaussianと同じく、全観測/filteredのordered sumとkept数を別々に持ち、
PR/CPそれぞれ `min(nDD,3)` に満たなければ全観測の尤度を選ぶ。
元のGPU FP64 sqrt・残差・AFV/rint・Huber正規化式を保つ。
初期実装はcost/weightの乗算とaccumulator加算の丸めが変わり451ケースでbit差が出たため棄却した。
PRは元のFMA、CPはweighted Huber costを `__dmul_rn` で丸めてから `__dadd_rn` で足す。
[棄却rawと初期source patch](pf_dd_huber_gate_rounding_rejected_20261012.json)を保存し、許容誤差を広げていない。
最終版は従来と**6,102ケースでstate/log weights/軌跡がbit一致**。

Huber enabled、PR/CP両kが正かつ有限、共通の有限参照座標がbit一致、
2〜511組DD、gate有効、有限観測、normal正波長の場合だけ新kernelを使う。
PRだけ/搬送波だけのHuber、mixed reference、gateなし、512組DD、特殊入力は従来の経路を使う。
新Huber kernelは**register59、stack/local/static shared0**。
dynamic sharedは12*nDD doubles（従来cached common jointの12*nDD+2から16byte減）で、
既存barrier1回・block256・allocation・H2D・launch数を保つ。
APIと既定設定は変更せず、既存の両Huber+gate設定で自動的に適用する。

### Gaussian gateでのpartial NaN修正

精度確認で、前回のGaussian単一loop版が `!(fabs(residual)>threshold)` をfiltered sumの条件に使い、
一部の距離計算だけoverflowしてNaNになる場合に、除外すべきNaN AFVを合計する不具合を確認した。
従来jointの正しい条件 `fabs(residual)<=threshold` に修正した。
4組中3組がgateを通り1組がoverflowするfixtureでは、旧Gaussian fusedはNaN、元のjointと修正版は**+0.0の同一bits**。
kept不足の場合は元の全観測fallbackを維持し、NaNもそのまま含める。
[修正前の実production出力](pf_dd_gate_partial_nan_baseline_20261012.json)、
[固定fixture](../tests/fixtures/pf_dd_gate_partial_nan.json)、
[62件の追加tests](../tests/test_pf_dd_huber_gate.py)で通常/片側gate/独立k/partial block/511・512境界/fallbackを検証した。
前回の極端値検査は全距離が同時に非有限になるケースを含んでいたが、
一部のみ非有限でkept数が足りるこのケースを含んでいなかった。

関連**1,280 tests pass**。追加62 testsをmemcheck・racecheck・synccheckでも順次実行し、errors/hazards/leaks0。
既存50 device関数のうち**49個のSASSが同一**で、変更した1個はGaussian filtered predicate修正。
Huber用1関数を追加し、[source/pyd/SASS/resource/CRLF再構成patch](pf_dd_huber_gate_device_code_20261012.json)を保存。
BVH本体とsignal batchのsource/build/installed SHAも維持した。
[最終整合確認](pf_dd_huber_gate_completion_check_20261012.json)。
CUDA 12.8 / MSVC Release / compute 7.5 / C++20 / no fast math。
SASS/resourceは静的確認で、occupancy実測やNsight結果は主張しない。

```powershell
python benchmarks/bench_pf_dd_huber_gate_compare.py --baseline dist/pf_dd_huber_gate_baseline/_gnss_gpu_pf_device.cp312-win_amd64.pyd --candidate build/diag/Release/_gnss_gpu_pf_device.cp312-win_amd64.pyd --baseline-dir dist/pf_dd_huber_gate_baseline --sizes 2048 10000 100000 1000000 --stages joint_huber_gate public_joint_huber_gate_step native_joint_huber_gate_step public_joint_gate_step joint_huber public_joint_huber_gate_mixed_step Student_control resample_control --rounds 5 --repeats 20 --output dist/pf_dd_huber_gate_repeat.json
```

## DD coefficient sharing, single-pass gates and epoch batching (2026-10-12)

精度を保つ3項目を個別の旧新production pydで評価して採用した。
共通参照DDのsigma係数共有は全PF stepで約0.7〜1.1%短縮、
Gaussianのゲート付きDD geometry統合は100万粒子の全stepで約17.6%短縮、
32epochの一括信号生成は公開APIで約45.0%、都市シミュレーション全体で約9.9%短縮。
各変更の直前版との比較であり、過去の速度比は掛け合わせない。

### GPU sigma coefficient sharing

| 処理全体の条件 | 旧 (ms) | 新 (ms) | 速度比 |
|:---|---:|---:|---:|
| 100万粒子 / split PR→CP 全step | 17.0937 | 16.9731 | 1.007× |
| 100万粒子 / joint 全step | 14.3918 | 14.2327 | 1.011× |

既存shared loadの中でblockのthread 0がGPU FP64の `1/(sigma*sigma)` を計算する。
PR/CPには8byte、jointには16byteのsharedを追加し、barrier・launch・allocationを増やさない。
共通参照経路に限定し、CPの非正・subnormal・非有限波長は従来のcommon kernelへ戻す。
jointの512組DDは48KiB上限のため従来の係数計算を保つ。
この小さな利得は両公開stepの全5組で確認したが、一律の利得は主張しない。
[32条件のraw](pf_dd_sigma_gpu_20261012.json)、[device audit/patch](pf_dd_sigma_device_code_20261012.json)。

### Single-pass Gaussian DD gates

| 処理全体の条件 | 旧 (ms) | 新 (ms) | 速度比 |
|:---|---:|---:|---:|
| 2,048粒子 / ゲート付きjoint 全step | 0.2481 | 0.2157 | 1.150× |
| 10,000粒子 / ゲート付きjoint 全step | 0.2978 | 0.2684 | 1.109× |
| 100,000粒子 / ゲート付きjoint 全step | 2.6820 | 2.1592 | 1.242× |
| 1,000,000粒子 / ゲート付きjoint 全step | 17.8731 | 14.7291 | 1.213× |

旧版はgateのkept数を決めるgeometryと尤度のgeometryを別々に計算した。
新版は同じ順序の1 loopで全観測とgate内観測の尤度を別々に蓄積し、
PR/CPそれぞれで `min(nDD,3)` のkept数によるfallbackを選ぶ。
PRのFMA、CPの乗算後の丸め・加算、AFV/rint、NaNのskip/count条件を維持した。
Gaussian・共通参照・512組未満・gate有効・有限観測・正のnormal波長だけ適用する。
Huber、mixed reference、例外観測は従来のinstruction pathを使う。
100万粒子の公開全stepは5組すべて1.210〜1.216×、native全stepも約17.4%短縮。
全stepはpredict/DD joint/ESS判定/条件付きB60 resample/estimateまで含む。
gate重み更新だけのroundや変更対象外のGaussian・resampleでは大きな揺れも観測したため、
単独kernel最大値や変更対象外の利得を採用根拠にしない。
[32条件のraw](pf_dd_gate_gpu_20261012.json)、[device audit/patch](pf_dd_gate_device_code_20261012.json)。

### Epoch batch signal generation

| 処理全体の条件 | 旧 (ms) | 新 (ms) | 速度比 |
|:---|---:|---:|---:|
| 公開生成 / 8epoch / 2,600 samples / 12channel | 1.6038 | 1.0984 | 1.460× |
| 公開生成 / 32epoch / 2,600 samples / 12channel | 7.5673 | 4.1618 | 1.818× |
| 公開生成 / 32epoch / 2,600 samples / 65channel | 30.2503 | 21.8600 | 1.384× |
| 公開生成 / 8epoch / 260,000 samples / 12channel | 29.5990 | 20.3392 | 1.455× |
| 都市全体 / 8epoch / 2,600 samples / 1,200三角形 | 6.8361 | 6.5048 | 1.051× |
| 都市全体 / 32epoch / 2,600 samples / 1,200三角形 | 26.0478 | 23.4727 | 1.110× |
| 都市全体 / 8epoch / 260,000 samples / 1,200三角形 | 30.0832 | 28.7919 | 1.045× |

新しい `SignalSimulator.generate_epochs(rows, n_samples, noise_seeds=None)` は
float32の `[epochs, 2*n_samples]` IQを返す。
channels・NAV・epoch descriptorを一括転送し、grid.yをepochにして1 launch・1 D2Hで処理する。
各epochは元のsample index/time・channel順・FP64 phase・float加算順・noise seedを使う。
固定seedと個別seedは従来の逐次生成とbit一致、seed 0はepochごとにfresh noise。
valid channel数が32以下/256以下/257以上の3経路、空row、未知system、NAVを照合した。
native/公開APIの時間は入力変換・packing・H2D・kernel・同期・返却IQまで含む。
都市全体の時間はLOS・反射・channel/scene metadata・trajectory yieldも含む。
単一epochのcontrolは0.1664→0.1746msで、既存の全5 device関数の命令列は一致した。
直接のray tracing高速化としては扱わない。

`UrbanSignalSimulator.simulate_trajectory(..., batch_size=32)` で有効にする。
既定は `batch_size=1` でepochごとのcallback/yieldを維持する。
一括時はchunkのcallbackを先に評価し、IQ chunkを32MiB以内に抑える
（1epoch自体が上限を超える場合は既存の単一epoch生成）。
native batchの上限は65,535epoch、同じsample数を要求する。
返却配列はworkspace再利用やrelease後も保持される。旧extensionへの逐次fallbackも用意した。
[13条件のraw](signal_batch_gpu_20261012.json)、[device audit/5ファイルpatch](signal_batch_device_code_20261012.json)。

### Precision and verification

sigmaの旧新版6,102ケース、gateの旧新版6,102ケースでstate・log weights・軌跡をbit照合。
さらに最終PFをsigma共有前と比較し、sigma²やgeometryのoverflow/underflow、zero/NaN粒子、
特殊波長を含む[360ケース](pf_dd_extreme_check_20261012.json)もbit一致。
signalは単一epochの457ケースへbatch/NAV/seed/都市metadataを加えた524ケースでbit一致。
計測したすべてのlog weights/返却値/IQ/metadataもhash照合した。
追加111 testsと関連1,218 testsがpass。
追加111 testsをmemcheck・racecheck・synccheckでも順次実行し、errors/hazards/leaksはいずれも0。
sigma前の46 instruction bodies、gate前の全49、signal前の全5を保持。
追加kernelはsigma PR44/CP42/joint51 registers、gate56、signal batch51/54、stack/local0。
SASS・resourceはcuobjdumpによる静的確認で、occupancyの実測やNsight結果は主張しない。

未ガード版はsigmaで48、gateで30ケースの特殊波長入力にbit差が出たため棄却した。
[sigmaの棄却結果/patch](pf_dd_sigma_unguarded_rejected_20261012.json)、
[gateの棄却結果/patch](pf_dd_gate_unguarded_rejected_20261012.json)を保存し、bit条件を緩めずguardを追加した。
rawには各phaseのsource/pyd/helper SHAと全round/sampleを保存。
device auditのpatchで各測定sourceのCRLF byteを再構成できる。
[source/build/installed・全成果物の最終整合確認](optimization_completion_check_20261012.json)。
baselineはそれぞれ前回joint、今回sigma、前回global-frequencyの測定SHAに固定。
実GPU処理は別processへ隔離して順次実行し、5交互round×20で比較した。
CUDA 12.8 / MSVC Release / compute 7.5 / C++20 / no fast math、hostは非占有。
初期化・state復元・BVH構築・IPC・出力hash照合は時間外。
BVH mirror-sharing本体（CUDA/header/bindings/bvh.py）のsource/build/installed SHAを保持。

再現時は各phaseのpatchでsourceを復元した個別checkoutを使い、対応するpydを使う。
sigmaの再現にはgate追加前のsourceが必要で、最終treeのsourceと古いsigma pydを混在させない。

```powershell
python benchmarks/bench_pf_dd_sigma_compare.py --baseline dist/pf_dd_sigma_baseline/_gnss_gpu_pf_device.cp312-win_amd64.pyd --candidate dist/pf_dd_gate_baseline/_gnss_gpu_pf_device.cp312-win_amd64.pyd --baseline-dir dist/pf_dd_sigma_baseline --sizes 2048 10000 100000 1000000 --stages PR_dd12 CP_dd12 joint_dd12 joint_gate public_CP_step public_joint_step Student_control resample_control --output dist/pf_dd_sigma_repeat.json
python benchmarks/bench_pf_dd_gate_compare.py --baseline dist/pf_dd_gate_baseline/_gnss_gpu_pf_device.cp312-win_amd64.pyd --candidate build/diag/Release/_gnss_gpu_pf_device.cp312-win_amd64.pyd --baseline-dir dist/pf_dd_gate_baseline --sizes 2048 10000 100000 1000000 --stages joint_gate public_joint_gate_step native_joint_gate_step joint_dd12 joint_huber joint_mixed Student_control resample_control --output dist/pf_dd_gate_repeat.json
python benchmarks/bench_signal_batch_compare.py --baseline dist/signal_batch_baseline/_gnss_gpu_signal_sim.cp312-win_amd64.pyd --candidate build/diag/Release/_gnss_gpu_signal_sim.cp312-win_amd64.pyd --baseline-dir dist/signal_batch_baseline --output dist/signal_batch_repeat.json
```

## PF DD joint reference reuse with preserved rounding (2026-10-12)

DD PR+搬送波の一括更新にも共通参照衛星の距離再利用を追加した。
100万粒子/12組DDの公開一括PF全stepは**16.3814→14.4195 ms（1.136×、約12.0%短縮）**。
別の5交互round×20でも**16.3794→14.4223 ms（1.136×、約11.9%短縮）**。
両測定とも全5組で改善し、対ごとの速度比は**1.135〜1.140× / 1.136〜1.139×**。
round中央値の範囲は本測定**旧 16.3696〜16.4244 ms / 新 14.4090〜14.4307 ms**、追試**旧 16.3255〜16.4817 ms / 新 14.3725〜14.4716 ms**。
共通参照の実測条件での利得であり、過去の速度比は掛け合わせない。

| 100万粒子の本測定条件 | 旧 (ms) | 新 (ms) | 速度比 |
|:---|---:|---:|---:|
| native joint / 12 pairs | 9.8299 | 7.4197 | 1.325× |
| native joint / 33 pairs | 20.0739 | 14.3925 | 1.395× |
| native joint gate / 12 pairs | 13.3041 | 9.2021 | 1.446× |
| native joint step / 12 pairs | 16.3725 | 14.3866 | 1.138× |
| public joint step / 12 pairs | 16.3814 | 14.4195 | 1.136× |
| unchanged public split PR→CP step | 17.1626 | 17.1396 | 1.001× |
| mixed-reference joint / original GPU body | 7.5314 | 7.5276 | 1.001× |
| unchanged Student control | 8.0986 | 8.0909 | 1.001× |
| unchanged B60 resample control | 3.3169 | 3.2266 | 1.028× |

全stepは予測/DD PR+CP一括重み付け/ESS判定/条件付きB60再サンプリング/推定を含む。
native単独重み付けにもhost packing/H2D/kernel/stream同期が含まれる。
初期化・state復元・出力照合・worker IPCは測定外。
実production pydを独立したpersistent workerへ読み込み、GPUを順次実行し、stageごとに旧新順を交互にした。
全timed log weights・返却値、各stage最終16状態はhash一致を確認した。
[最終版60条件・4粒子数・5×20](pf_dd_joint_reference_gpu_20261012.json)、
[最終版の追加5条件・5×20](pf_dd_joint_reference_check_20261012.json)。

### 丸めを保つ実装と適用条件

前回のbit差は、奇数組のGaussian CP尤度の末尾で乗算と加算が融合する箇所と対応した。
共通参照版の最終CP乗算を `__dmul_rn` で明示し、旧版の乗算後の丸めを保つ。
さらにPR/CPの個別設定を照合すると、搬送波だけにHuberを使う構成で12ケースにbit差が出た。
この構成（Huber enabled、PR kが正でない、CP kが正）は元のjointカーネルへdispatchする。
許容誤差は広げず、既存の丸めを保存できる構成だけ距離共有を使う。
[個別設定の棄却結果](pf_dd_joint_carrier_huber_rejected_20261012.json)と
[ガード前の再現patch](probes/pf_dd_joint_carrier_huber_rejected_20261012.patch)を保持した。

DD2組以上、参照ECEFの全3座標が有限、全組の参照座標がbit一致し、上記の構成でない場合に適用する。
毎呼び出し、既存shared load後に元のGPU FP64 sqrt式で粒子ごとの参照距離を1回計算する。
各組の衛星距離、残差、AFV/rint、独立したPR/CP gate・fallback、Huber、ordered sum、上書き更新は維持する。
mixed/1 ulp差/signed zero差/nonfinite/single-DDも元のGPU計算を使い、呼び出し間のgeometry cacheは持たない。
前回の単独PR/CP距離再利用とinverse-wavelength共有、API、allocation、H2D byte数、launch数、
block256、既存barrier1回、dynamic shared 12*n_dd doublesはそのまま。

mixed jointの参照差分とsqrtのソース順も旧版通りに残し、
**既存全46関数のSASS instruction bodyが一致**した。共通joint用に1関数を追加。
共通版は**register51（元のjoint48）/stack・local・static shared0**。
occupancyの実測値や全構成への一律の利得は主張しない。
[device audit](pf_dd_joint_reference_device_code_20261012.json)にsource/pyd/resource/SASS/patchを固定。
BVH mirror-sharing・signal global-frequencyのsource/build/installed pydも既存SHAを保持した。

### 精度と回帰防止

前回4,338ケースへ奇数DD・3 seed・3組のsigma・座標の極端値など1,044ケースを追加し、
最終版も**5,382ケースの16状態・log weights・3 epoch trajectoryが旧新版bit一致**。
個別gate4組合せ、Huber6組合せ、DD3/12/33/511、1/31/257粒子の
**追加288ケースも旧新版bit一致**した。
[個別設定checker](bench_pf_dd_joint_options_check.py)、[個別設定結果](pf_dd_joint_options_check_20261012.json)。
計**5,670ケース**で丸めの保存を確認した。

前回不一致の `N256/dd33/mixed/gaussian/joint` の粒子239を、旧版・棄却版の実pydで再現した。
input/output hashを前回記録と照合し、**2 ulpの重み差**を
[fixture](../tests/fixtures/pf_dd_joint_rounding.json)へ保存。
1/31/257粒子で正確なuint64期待値と比較する。
さらにsigned zeroだけ異なる参照で元のmixed kernelを使い、同じ距離の独立GPU計算とbit照合した。
partial blocks、最大48KiB shared、参照更新・buffer成長、carrier-only Huberの元の計算も検証。
**881関連テストPASS、新72テストのmemcheck/racecheck/synccheck全て0、leak0**。
前回の棄却[raw](pf_dd_reference_joint_rejected_20261012.json)・[patch](probes/pf_dd_reference_joint_rejected_20261012.patch)も保持した。

### 測定範囲と対照条件

| 100万粒子の追試条件 | 旧 (ms) | 新 (ms) | 速度比 |
|:---|---:|---:|---:|
| native joint / 12 pairs | 9.7279 | 7.9489 | 1.224× |
| public common-reference joint step | 16.3794 | 14.4223 | 1.136× |
| public mixed-reference joint step | 16.3456 | 16.3400 | 1.000× |
| unchanged Student control | 10.6160 | 12.8759 | 0.824× |
| unchanged B60 resample control | 3.3171 | 3.9703 | 0.835× |

host/GPUは非専有。未変更の対照処理も測定ごとに変動するため、時間を補正したりrawを置き換えたりしない。
混在参照のGPU命令列は旧版と同一で、今回の距離再利用の対象は共通参照のみ。
小粒子数・DD組数・全Huber構成への一律の高速化は主張せず、公開共通参照PF全stepの繰り返し改善を採用根拠にした。
ガード前の計測・pyd・sourceは最終版の数値へ混ぜず `dist/pf_dd_joint_rounding_probe/` に別保存した。

[source patch](probes/pf_dd_joint_reference_20261012.patch)で前回のPR/CP参照距離共有版から
最終測定したCRLF sourceをbyte単位で復元できることを確認した。
[benchmark](bench_pf_dd_joint_reference_compare.py)、[テスト](../tests/test_pf_dd_joint_reference.py)。再現:

```powershell
python benchmarks/bench_pf_dd_joint_reference_compare.py --baseline dist/pf_dd_joint_baseline/_gnss_gpu_pf_device.cp312-win_amd64.pyd --candidate build/diag/Release/_gnss_gpu_pf_device.cp312-win_amd64.pyd --baseline-dir dist/pf_dd_joint_baseline --rounds 5 --repeats 20 --output dist/pf_dd_joint_final_20261012.json
python benchmarks/bench_pf_dd_joint_reference_compare.py --baseline dist/pf_dd_joint_baseline/_gnss_gpu_pf_device.cp312-win_amd64.pyd --candidate build/diag/Release/_gnss_gpu_pf_device.cp312-win_amd64.pyd --baseline-dir dist/pf_dd_joint_baseline --sizes 1000000 --stages joint_dd12 public_joint_step public_joint_mixed Student_control resample_control --rounds 5 --repeats 20 --output dist/pf_dd_joint_check_20261012.json
python benchmarks/bench_pf_dd_joint_options_check.py --baseline dist/pf_dd_joint_baseline/_gnss_gpu_pf_device.cp312-win_amd64.pyd --candidate build/diag/Release/_gnss_gpu_pf_device.cp312-win_amd64.pyd --output dist/pf_dd_joint_options_check_20261012.json
```

## PF DD common reference-range reuse (2026-10-12)

共通参照衛星を持つDD PR・搬送波の単独更新で、粒子から参照衛星までの距離の重複計算を省いた。
100万粒子/12組DDの**公開DD PR→搬送波の分割PF stepは21.0943→17.1320 ms
（1.231×、約18.8%短縮）**。全5交互roundで改善し、対ごとの速度比は**1.222〜1.232×**。
旧round中央値の範囲**21.0559〜21.1398 ms**と新**17.1184〜17.2314 ms**は重ならない。
先行した3交互round×5でも**20.8576→16.9632 ms（1.230×、約18.7%短縮）**で全3対が改善。
共通参照の分割更新で得た利得であり、過去の速度比は掛け合わせない。

| 100万粒子の条件 | 旧 (ms) | 新 (ms) | 速度比 |
|:---|---:|---:|---:|
| native DD PR / 12 pairs | 7.3769 | 5.0648 | 1.457× |
| native DD CP / 12 pairs | 6.9714 | 4.6903 | 1.486× |
| native DD PR / 32 pairs | 15.3212 | 9.8616 | 1.554× |
| native DD CP / 32 pairs | 16.3495 | 10.8167 | 1.511× |
| native DD PR→CP step / 12 pairs | 20.9385 | 17.0477 | 1.228× |
| public DD PR→CP step / 12 pairs | 21.0943 | 17.1320 | 1.231× |
| unchanged public joint step / 12 pairs | 16.3775 | 16.3815 | 1.000× |
| unchanged Student control | 8.0972 | 8.1075 | 0.999× |
| unchanged B60 resample control | 3.3123 | 3.1516 | 1.051× |

stepは予測/DD PR/DD搬送波AFV/ESS判定/条件付きB60再サンプリング/推定を含む。
native単独重み付けにもhost packing/H2D/kernel/stream同期が含まれる。
初期化・state復元・出力照合・worker IPCは測定外。
全timed log weights・返却値とstage最終16状態を旧新版でhash照合した。
**96条件、4粒子数（2,048/1万/10万/100万）、5交互round×20**の
[本測定](pf_dd_reference_gpu_20261012.json)と、
[先行6条件・3交互round×5](pf_dd_reference_probe_20261012.json)を保存した。

### 適用条件と精度

毎呼び出しのhost入力で、DDが2組以上、参照ECEFの3座標が有限、
全組の参照座標がbit一致することを検査する。
この場合のみ、元のGPU FP64 sqrt式を粒子ごとに1回計算し、gate・尤度ループで再利用する。
観測を更新するたびに距離を再計算し、呼び出し間のgeometry cacheは持たない。
混在参照、1 ulpだけ異なる参照、signed zeroの違い、非有限参照、単独DDは元の計算経路を使う。
残差/AFV/rint/gateのfallback/Huber/ordered sumは同じ。
DD PRの上書きとDD CPの加算、前回のGPU inverse-wavelength共有も維持した。

API、owner allocation、H2D byte数、kernel launch数、block256、既存barrier1回、
dynamic shared（PR 10*n_dd / CP 11*n_dd doubles）は変更しない。
共通参照カーネルは**PR register44（旧44）、CP42（旧48）、stack/local/static shared0**。
既存の**全44関数のSASS instruction bodyは一致**し、共通参照用に2関数を追加。
mixed-reference templateの関数名は変わるが、instruction body自体は旧単独カーネルと一致する。
[device audit](pf_dd_reference_device_code_20261012.json)にsource/pyd/resource/SASS/patchを固定した。
BVH mirror-sharing・signal global-frequencyのsource/build/installed pydも既存SHAを保持。

**4,338ケース**で16状態・log weights・3 epoch trajectoryの旧新版がbit一致。
元の波長検証2,316ケースに、PR/CP/joint、共通/混在/1 ulp差/signed zero/原点/巨大/非有限参照、
partial blocks、DD1/2/12/512、gate/Huber、参照更新とbuffer成長を加えた。
さらに**144ケース**で粒子座標の原点、signed zero、NaN/Inf、巨大値、subnormalを含めてbit一致。
[追加checker](bench_pf_dd_reference_precision_check.py)、[追加結果](pf_dd_reference_precision_check_20261012.json)。
新50テストはsigned zeroだけ異なる参照で元のmixed kernelを使い、同じ距離の独立GPU計算とbit比較する。
**809関連テストPASS、新50テストのmemcheck/racecheck/synccheck全て0、leak0**。

### 見送った一括更新と測定の限界

最初はDD jointにも同じ再利用を試したが、4,338ケース中**38ケースで旧新版のbit不一致**が出たため採用しなかった。
許容誤差は広げず、jointのsource・呼び出し・GPU命令列を元に戻した。
[不一致の入力/出力hash](pf_dd_reference_joint_rejected_20261012.json)、
[試作patch](probes/pf_dd_reference_joint_rejected_20261012.patch)、
[試作runner](probes/bench_pf_dd_reference_joint_probe_20261012.py)を保存。
採用版の一括更新の公開stepは**1.000×**で、今回の利得は分割更新に限る。

host/GPUは非専有。100万粒子の未変更Studentは**0.999×**、B60 resampleは**1.051×**、
混在参照PR/CPも**1.041×/1.023×**と変動した。
2,048粒子の未変更public jointにも**1.277×**が出ており、こうした値を最適化の利得と扱わない。
100万粒子/2組CPは中央値**1.094×**だが対ごとの1 roundは**0.889×**。
全粒子数・観測数への一律の改善は主張せず、最大値より分割PF全stepの繰り返し改善を採用根拠とした。

[source patch](probes/pf_dd_reference_range_20261012.patch)は直前のinverse-wavelength版から
測定したCRLF sourceをbyte単位で復元できることを確認した。
[benchmark](bench_pf_dd_reference_compare.py)、[テスト](../tests/test_pf_dd_reference_range.py)。再現:

```powershell
python benchmarks/bench_pf_dd_reference_compare.py --baseline dist/pf_dd_reference_baseline/_gnss_gpu_pf_device.cp312-win_amd64.pyd --candidate build/diag/Release/_gnss_gpu_pf_device.cp312-win_amd64.pyd --baseline-dir dist/pf_dd_reference_baseline --rounds 5 --repeats 20 --output dist/pf_dd_reference_final_20261012.json
python benchmarks/bench_pf_dd_reference_precision_check.py --baseline dist/pf_dd_reference_baseline/_gnss_gpu_pf_device.cp312-win_amd64.pyd --candidate build/diag/Release/_gnss_gpu_pf_device.cp312-win_amd64.pyd --output dist/pf_dd_reference_precision_check_20261012.json
```

## PF DD carrier inverse-wavelength cache (2026-10-12)

DD搬送波AFVとDD PR+搬送波一括更新で、粒子ごとの `1.0 / wavelength` の重複計算を省いた。
100万粒子/12組のDD観測の公開一括PF stepは **17.2490→16.4796 ms
（1.047×、約4.5%短縮）**。
別の5交互round×20でも **17.0426→16.3059 ms（1.045×、約4.3%短縮）**。
両測定とも全5 roundで改善し、速度比は **1.041〜1.048× / 1.043〜1.048×**。
実測条件での全step利得であり、過去の速度比は掛け合わせない。

### 同期と実PF更新を含む呼び出し

| 100万粒子の条件 | 旧 (ms) | 新 (ms) | 速度比 |
|:---|---:|---:|---:|
| native DD CP / 12 pairs | 8.5398 | 7.4248 | 1.150× |
| native DD CP / 32 pairs | 18.3627 | 16.3800 | 1.121× |
| native DD joint / 32 pairs | 21.4494 | 19.4847 | 1.101× |
| native DD CP step / 12 pairs | 21.7308 | 21.0203 | 1.034× |
| native DD joint step / 12 pairs | 17.0602 | 16.3494 | 1.043× |
| public DD CP step / 12 pairs | 21.8406 | 21.1264 | 1.034× |
| public DD joint step / 12 pairs | 17.2490 | 16.4796 | 1.047× |

stepは予測/DD PR・搬送波/ESS判定/条件付きB60再サンプリング/推定を含む。
CP stepはDD PRとDD CPを別呼び出し、joint stepは既存の一括APIを使用。
native重み付けにもhost packing/H2D/kernel/stream同期が含まれる。
初期化・state復元・出力hash・worker IPCは測定外。
固定入力の全timed log-weight配列と返却値をhash確認し、各stageの最終16状態も一致を確認した。

### 変動の大きかった条件の追加確認

| 100万粒子の追試条件 | 旧 (ms) | 新 (ms) | 速度比 |
|:---|---:|---:|---:|
| native DD CP / 12 pairs | 7.2740 | 6.3171 | 1.151× |
| native DD joint / 12 pairs | 8.2942 | 7.5652 | 1.096× |
| native DD joint gate / 12 pairs | 14.8162 | 13.3650 | 1.109× |
| public DD joint step / 12 pairs | 17.0426 | 16.3059 | 1.045× |
| unchanged Student control | 8.0053 | 8.0219 | 0.998× |
| unchanged B60 resample control | 3.1834 | 3.2376 | 0.983× |

本測定の未変更Student/resampleは **0.817× / 0.686×** まで変動したため、7条件を追加確認した。
両方のrawを残し、本測定を追試値で置き換えない。
native joint12は本測定 **0.988×** に対し追試 **1.096×**、
joint gateは本測定 **1.844×** に対し追試 **1.109×**。
CP4も本測定 **1.464×** から追試 **0.925×** へ変わり、最大値を採用根拠にしない。
12 pairsの公開joint全stepは両測定で旧・新のround中央値の範囲が重ならず、
本測定旧 **17.1523〜17.2623** /新 **16.4641〜16.5086 ms**、
追試旧 **17.0131〜17.0554** /新 **16.2803〜16.3209 ms**。
native CP12の中央値は両方 **1.15×** だが、追試のround対1回は **0.984×**。
host/GPU非専有であり、全観測数・サイズに一律の利得は主張しない。
[本測定64条件](pf_dd_wavelength_gpu_20261012.json)、[追試7条件](pf_dd_wavelength_check_20261012.json)。

既存のshared観測loadの最終columnを、raw wavelengthから元のGPU FP64 reciprocalに置き換える。
DD CPの11-column、DD jointの12-column、総sharedサイズ、既存barrier1回、block256、launch数、
host packing、H2D byte数、owner allocation、APIは同じ。
GPUで毎block・毎呼び出しに `1.0 / wavelength` を計算し、gateループと尤度ループが同じ値を参照する。
CPUでの係数計算・近似逆数・残差式/rint/Huber/gate/ordered accumulationの変更はない。
CPの加算更新とjointの上書き更新も維持し、波長やDD数を変えたときも新しい入力から準備する。
2カーネルとも**register48（旧48）/static shared・stack・local0**。
他の**42/44関数のSASS instruction bodyが全一致**。
[device audit](pf_dd_wavelength_device_code_20261012.json)へsource/pyd/resource/SASS/patchを保存。
BVH mirror-sharing版とsignal global-frequency版のsource/build/installed pydはSHA一致で保持した。

直前Student版のsource5件と実pydへbaseline固定。独立したpersistent production workerを使い、
GPU順次・stageごと5交互round×20で測定。
**2,316ケース**の16状態・log weights・3 epoch trajectoryが旧新版でbit一致。
partial blocks、DD1/2/4/12/32/33/257/512、mixed/constant/tiny/huge/nonfinite波長、zero weights、
gate/Huber、観測buffer成長、100万粒子を含む。
**追加16条件**でもCP/joint/splitそれぞれの旧新版がbit一致。
一括と分割の間には旧版からFMA丸め差があり、同じkernel間の旧新版照合と混同しない。
数値比較のテストではこの既存差を許容し、
[追加precision checker](bench_pf_dd_wavelength_gate_check.py)と
[比較結果](pf_dd_wavelength_gate_check_20261012.json)で各経路のbit保存を別途確認した。
さらに未変更の単独AFV/単独DD PRを用いる独立照合と、最大48KiB shared・partial block・更新を検証。
**759関連テストPASS**、新59テストのmemcheck/racecheck/synccheckすべて0、leak0。

[source patch](probes/pf_dd_wavelength_cache_20261012.patch)は直前測定sourceから今回の
CRLF sourceをbyte単位で復元できることを確認した。
[benchmark](bench_pf_dd_wavelength_compare.py)、[追加7条件runner](bench_pf_dd_wavelength_check.py)、
[テスト](../tests/test_pf_dd_wavelength_cache.py)。再現コマンド:

```powershell
python benchmarks/bench_pf_dd_wavelength_compare.py --baseline dist/pf_dd_wavelength_baseline/_gnss_gpu_pf_device.cp312-win_amd64.pyd --candidate build/diag/Release/_gnss_gpu_pf_device.cp312-win_amd64.pyd --baseline-dir dist/pf_dd_wavelength_baseline --rounds 5 --repeats 20 --output dist/pf_dd_wavelength_final_20261012.json
python benchmarks/bench_pf_dd_wavelength_check.py --baseline dist/pf_dd_wavelength_baseline/_gnss_gpu_pf_device.cp312-win_amd64.pyd --candidate build/diag/Release/_gnss_gpu_pf_device.cp312-win_amd64.pyd --baseline-dir dist/pf_dd_wavelength_baseline --rounds 5 --repeats 20 --output dist/pf_dd_wavelength_check_20261012.json
```

## BVH selection, tiling and distance probes — not adopted (2026-10-12)

実効的な改善の確認を優先し、3案を実ビルド・順次3交互round×10で探索した。
全timed出力をhash/bit確認したが、大きな呼び出し全体で十分な利得がなく、**3案とも不採用**。
直前の受信機鏡像点共有版へCUDA sourceとbuild/installed pydを**byte一致で復元**した。
古いobject/libraryが残らないよう復元sourceを再ビルドした後、測定済みのpyd bytesを戻している。

| 試案 / 主な比較条件 | 旧 (ms) | 試案 (ms) | 速度比 |
|:---|---:|---:|---:|
| warp選択 / 公開12,000三角形・192 rays | 3.4601 | 3.4809 | 0.994× |
| warp選択 / 公開512三角形・4,096 rays | 3.0528 | 3.0264 | 1.009× |
| tile上限解除 / 公開120,000三角形・1 ray | 1.0638 | 1.1262 | 0.945× |
| tile上限解除 / 公開120,000三角形・2 rays | 1.7817 | 1.8833 | 0.946× |
| 距離共有 / 公開12,000三角形・192 rays | 3.3541 | 3.3598 | 0.998× |
| 距離共有 / 公開512三角形・4,096 rays | 2.9276 | 2.8532 | 1.026× |

ray検証/H2D/kernel/同期/D2H/返却まで含み、CPU BVH構築・生成・出力hashは測定外。
coldは構築済みCPU meshからowner初期化・準備・closeも含む。
host/GPU非専有の探索で、未変更LOSにも変動がある。3回の探索中央値を安定した利得とみなさず、
正式採用版の5×20や過去の速度比と混ぜない。

warp案はdelay/visit ordinalだけをshuffle比較し、勝ったthreadだけがpayloadを書いた。
共有領域 **5120→160 bytes**、barrier **8→1**、一方でcandidate registersは
uncached/normal-cache **78→82**、mirror共有 **77→78**。132検証ケースはbit一致。
28 timed条件を測定し、上の大きなbatch全体ではほぼ利得がなかったため戻した。
[raw結果](bvh_warp_selection_rejected_20261012.json)、
[patch](probes/bvh_warp_selection_rejected_20261012.patch)、[runner](bench_bvh_selection_compare.py)。

tile案は独立候補の32-tile上限を外し、従来の256-block目標に合わせた。
GPU **全9関数のSASS instruction bodyが一致**、158ケースもbit一致だったが、
1〜2 raysでは呼び出し全体が低下した。多いtilesの準備・最終選択を含む費用を回収できず不採用。
原因をGPU profilerで確定したとは主張しない。32 timed条件を保存。
[raw結果](bvh_tiles_rejected_20261012.json)、
[patch](probes/bvh_tiles_rejected_20261012.patch)、[runner](bench_bvh_tiles_compare.py)。

距離案は同一rayのFP64距離平方根をthread0で一度評価し、shared doubleから読んだ。
共有領域 **5120→5128 bytes**、barrier **8→9**、registersはuncached/normal-cache **76**、
mirror共有 **78**。158ケースbit一致、28 timed条件で大きなbatch全体はほぼ不変だった。
512三角形/4,096 raysの約2.6%だけを根拠に採用せず、元の経路へ戻した。
[raw結果](bvh_distance_rejected_20261012.json)、
[patch](probes/bvh_distance_rejected_20261012.patch)、[runner](bench_bvh_distance_compare.py)。

[device audit](bvh_probes_device_code_20261012.json)にsource/pyd/SASS/resource/patch再現情報を保存。
各patchは同じ直前のmirror-sharing sourceから測定CRLF sourceをbyte単位で再現できる。
復元後は**484関連テストPASS**。PF/signalも直前sourceとbuild/installed binaryへSHA一致で保持。
不採用の探索案にsanitizerの実施結果を主張せず、直前採用版の結果を流用しない。

再現には記録したbaselineとpatchから該当案を別途ビルドし、対応runnerを使う。
探索の主な引数はwarp/distanceが`--counts 512 1200 12000 --rounds 3 --repeats 10 --skip-urban --skip-controls`、
tilesが`--counts 12000 120000 --rays 1 2 4 8 --rounds 3 --repeats 10 --skip-urban --skip-controls`。
baseline-dirは`dist/bvh_selection_baseline`。探索時のsource・binaryはignored distの個別archiveにも残した。

## Resident BVH receiver mirror sharing (2026-10-12)

同じepochの衛星は三角形ごとに同じ受信機鏡像点を使うため、GPUで一度計算して共有した。
対象を**訪問三角形8,192以上・epochあたり16衛星以上・全体4,096 rays以下**に絞り、
追加scratchが**32 MiB以下**の場合に使う。12衛星の都市処理は直前の法線cache経路を保つ。
120,000三角形/192 raysの公開LOS・反射全体は **32.0752→28.7006 ms
（1.118×、約10.5%短縮）**、12,000三角形/192 raysは **3.7584→3.4069 ms（1.103×）**。
両条件とも5 roundすべてで改善し、速度比はそれぞれ **1.115〜1.118× / 1.097〜1.105×**。
効果の中心は大きなレイバッチ。過去の速度比とは掛け合わせない。

### 転送・同期・返却配列を含む呼び出し

| 条件 | 旧 (ms) | 新 (ms) | 速度比 |
|:---|---:|---:|---:|
| 反射 / 12,000三角形 / 192 rays | 3.2129 | 2.8761 | 1.117× |
| 公開LOS・反射 / 12,000 / 192 rays | 3.7584 | 3.4069 | 1.103× |
| 反射 / 120,000 / 192 rays | 30.1022 | 26.7656 | 1.125× |
| 公開LOS・反射 / 120,000 / 192 rays | 32.0752 | 28.7006 | 1.118× |
| 初回生成・trace・close / 12,000 / 192 rays | 5.4326 | 5.1305 | 1.059× |
| 初回生成・trace・close / 120,000 / 192 rays | 49.3223 | 45.0541 | 1.095× |

反射はnative `DeviceBVH.multipath_batch`、公開LOS・反射はPython `BVHAccelerator.trace_batch`。
ray検証/H2D/鏡像点の毎回の準備/全kernel/同期/D2H/返却配列を含み、kernel単体の時間ではない。
初回呼び出しは構築済みCPU nodes/trianglesからowner生成/upload/法線準備/trace/closeも含む。
メッシュ生成・CPU BVH構築・出力hashは測定外。

### 32衛星の都市シミュレーション全体

| 三角形 / samples | 旧 (ms) | 新 (ms) | 速度比 |
|:---|---:|---:|---:|
| 1,200 / 2,600 | 3.1296 | 3.1788 | 0.984× |
| 1,200 / 260,000 | 9.4391 | 9.7210 | 0.971× |
| 12,000 / 2,600 | 7.4697 | 7.4882 | 0.998× |
| 12,000 / 260,000 | 10.8163 | 10.5981 | 1.021× |
| 120,000 / 2,600 | 7.2760 | 6.7926 | 1.071× |
| 120,000 / 260,000 | 10.2990 | 9.8646 | 1.044× |

LOS/反射/Python channel構築/実CUDA IQ/同期・返却まで測定。
120,000三角形/260,000 samplesは **1.044×（約4.2%短縮）**。round対は全5回改善したが、
速度比 **1.005〜1.066×**、baseline round中央値 **8.6014〜12.2681 ms** と変動が大きい。
都市全体に一律の利得を主張せず、上記の大きなレイバッチを主な採用根拠とする。
1,200三角形/32衛星は未変更経路で、26万samplesでも **0.971×**。
未変更12衛星の都市6条件は **0.948〜1.047×**、LOS-only全10条件は **0.988〜1.018×**。
未変更4,097-ray serialは反射 **1.000×** / trace **1.001×**。
host/GPU非専有の実測で、未変更経路の差は新kernelの効果として数えない。
全timed IQ・channel metadataもbit/hash一致を確認した。

GPUの鏡像点準備kernelはepoch×訪問三角形を担当し、元のFP64
`rx - v0` / dot / `rx - 2*d*n` の式を評価する。
受信機依存の値は**呼び出しごとに再計算**し、同じdefault streamの後続反射kernelが参照する。
方向の正規化・Moller-Trumbore交差式・距離・同遅延のordinal比較は同じ。
入力の変更やsatellite数・epoch数・fallbackの切り替えでも古い鏡像点を使わない。
stateless API、小メッシュ、15衛星以下、serial、scratch上限超過は直前の計算経路へ戻る。
LOS-onlyでは鏡像点bufferを確保しない。

追加scratchは**24 bytes×訪問三角形数×epoch数**、確保量は32 MiBを超えない。
今回の1 epochでは12,000三角形で **288,000 bytes**、120,000で **2,880,000 bytes** 増加。
同owner内では最大使用量まで再利用し、`allocated_bytes`に計上、`close`/`release_gpu`で解放する。
準備kernelは**register30/shared・stack・local0**、共有candidateは
**register77（直前78）/shared5120 bytes/stack・local0**。
直前の7関数（法線準備とnormal-cache候補を含む）の**SASS instruction bodyが全一致**。
[device audit](bvh_mirror_device_code_20261012.json)にresource/hash/dispatch/patch検証を保存した。
PF/signalの直前sourceとbuild/installed binaryもSHA一致で保持した。

直前法線cacheのsource5件と実pydへbaseline固定。GPU順次・5交互round×20、全**54 timed条件**の
raw samples/round中央値/全出力hashを[測定JSON](bvh_mirror_gpu_20261012.json)に保存。
**132 LOS/反射検証ケース**でbit一致。
既存random/ECEF/tie/degenerate/scaleに加え、8191/8192/8193三角形、15/16/17衛星、
32 MiB上限の170/171 epochs、shared鏡像点でのtie/nonfinite、受信機の連続変更を確認した。
**484関連テストPASS**、新25件のmemcheck/racecheck/synccheckすべて0、leak0。
[再現patch](probes/bvh_receiver_mirror_sharing_20261012.patch)は直前sourceから測定CRLF sourceを
byte単位で復元できる。
[runner](bench_bvh_mirror_compare.py)、[更新・境界テスト](../tests/test_bvh_mirror_sharing.py)。

```powershell
python benchmarks/bench_bvh_mirror_compare.py --baseline dist/bvh_geometry_baseline/_bvh.cp312-win_amd64.pyd --candidate build/diag/Release/_bvh.cp312-win_amd64.pyd --baseline-dir dist/bvh_geometry_baseline --rounds 5 --repeats 20 --output dist/bvh_mirror_adopt_20261012.json
```

### 探索で見送った範囲

辺/v0/法線を104-byte recordにまとめる案は110ケースbit一致、3交互round×10の探索で
120,000三角形/192 raysの反射が約1.075×だったが、法線cache32 bytesの**3.25倍の容量**と
初回コストに対して全体利得が安定しなかったため見送った。
[raw探索結果](bvh_geometry_rejected_20261012.json)、[patch](probes/bvh_geometry_rejected_20261012.patch)、
[runner](bench_bvh_geometry_compare.py)を保存し、本番sourceへ取り込んでいない。
これは探索比較であり、採用版の484テスト・sanitizer結果をこの不採用版へ流用しない。

鏡像点共有を8衛星から使う試案も132ケースbit一致・5交互round×20で測定した。
12,000三角形/12衛星の都市26万samplesは **0.997×**、120,000/12衛星は **0.984×**で、
初回処理も低下したため採用版は16衛星以上へ絞った。
[raw dispatch探索](bvh_mirror_dispatch_probe_20261012.json)、
[source patch](probes/bvh_mirror_small_group_probe_20261012.patch)、
[当時のrunner snapshot](probes/bench_bvh_mirror_small_group_probe_20261012.py)を保存。
snapshotのimportには`PYTHONPATH=benchmarks`を使い、raw報告のrunner SHAと一致することを確認した。
両試案のpatchも直前法線cacheから測定sourceをbyte単位で復元できる。

## Resident BVH reflection normal cache (2026-10-12)

常駐メッシュの三角形法線をGPUで一度だけ計算し、レイごとの重複したcross・sqrt・FP64正規化を省いた。
12,000三角形/192 raysの公開LOS・反射一括呼び出しは **5.6700→3.7431 ms（1.515×）**。
120,000三角形/12衛星/260,000 samplesの都市シミュレーション全体は
**5.4601→3.9921 ms（1.368×、約26.9%短縮）**。
この都市条件のround対は5回とも改善し、速度比は **1.321〜1.407×**。
host/GPU非専有での実測条件の結果であり、過去の速度比とは掛け合わせない。

### 転送・同期・返却配列を含む常駐呼び出し

| 条件 | 旧 (ms) | 新 (ms) | 速度比 |
|:---|---:|---:|---:|
| 反射 / 12,000三角形 / 12 rays | 0.4135 | 0.2601 | 1.590× |
| 反射 / 12,000 / 192 rays | 5.1278 | 3.2043 | 1.600× |
| 公開LOS・反射 / 12,000 / 192 rays | 5.6700 | 3.7431 | 1.515× |
| 公開LOS・反射 / 120,000 / 12 rays | 3.7899 | 2.3253 | 1.630× |
| 公開LOS・反射 / 120,000 / 192 rays | 51.7678 | 32.0369 | 1.616× |

反射はnative `DeviceBVH.multipath_batch`、公開LOS・反射はPython `BVHAccelerator.trace_batch`。
CUDA kernelだけの時間ではなく、ray検証/H2D/全kernel/同期/D2H/返却配列を含む。
メッシュ生成・CPU BVH構築・出力hashは測定外。
12,000三角形/192 raysの公開一括呼び出しは5 roundとも **1.513〜1.519×**。

### 初回生成・反射準備・解放を含むnative呼び出し

| 三角形 / rays | 旧 (ms) | 新 (ms) | 速度比 |
|:---|---:|---:|---:|
| 512 / 12 | 0.3120 | 0.3162 | 0.987× |
| 1,200 / 12 | 0.3426 | 0.3442 | 0.995× |
| 12,000 / 192 | 7.2152 | 5.2519 | 1.374× |
| 120,000 / 12 | 18.8858 | 17.8865 | 1.056× |
| 120,000 / 192 | 65.9361 | 47.0323 | 1.402× |

CPU構築済みの同じnodes/trianglesから、owner生成/upload/初回order・法線準備/trace/closeを計測。
小規模・少数レイでは追加malloc・kernelのコストを回収しきれず、一律に改善しない。
12三角形の未変更経路は公開12 rays **0.971×**、192 rays **1.002×**。
未変更LOS-onlyの全10条件は **0.988〜1.010×**、未変更serial fallback/4,097 raysは
反射 **1.002×**、trace **1.000×**。

### 12衛星の都市シミュレーション全体

| 三角形 / samples | 旧 (ms) | 新 (ms) | 速度比 |
|:---|---:|---:|---:|
| 1,200 / 2,600 | 0.5625 | 0.5461 | 1.030× |
| 1,200 / 260,000 | 1.6360 | 1.5891 | 1.029× |
| 12,000 / 2,600 | 1.9120 | 1.5416 | 1.240× |
| 12,000 / 260,000 | 2.7971 | 2.5269 | 1.107× |
| 120,000 / 2,600 | 4.4522 | 2.9790 | 1.495× |
| 120,000 / 260,000 | 5.4601 | 3.9921 | 1.368× |

LOS/反射/Python channel構築/実CUDA IQ/同期・返却を含み、CPU BVH構築は測定外。
両版に直前のsignal global-frequency cacheを同じsource/binaryで使用した。
IQと全channel metadataも各timed callで一致を確認。

最初のparallel反射呼び出しで、訪問順に `ReflectionNormal` をGPU上で用意する。
元のFP64 cross/sqrt/3 divisionとstrict cutoff `nn < 1e-15` を使い、同じdefault streamで
準備後に反射kernelを実行する。CPUでの法線計算、近似逆数、反射式・演算順・比較順の変更はない。
以後は既存の2 reflection kernelsを使い、同じ法線を参照する。
訪問ordinalによる同遅延のtie選択、深い木のstack制限、非有限rayのzero結果も維持する。
stateless API・512未満のメッシュ・4,097 rays以上のserial経路は元の計算を使う。
LOS-onlyとserialのみ利用するownerでは法線cacheのallocation/準備kernelを行わない。

追加常駐メモリは**訪問三角形あたり32 bytes**。
今回の12,000三角形では **384,000 bytes**、120,000では **3,840,000 bytes** 増え、
`allocated_bytes` に計上される。`close`/`release_gpu`で解放し、mesh更新では新ownerが再計算する。
cached candidateは**register78/shared5120 bytes/stack・local0**（旧candidateと同じ）。
法線準備は**register36/shared・stack・local0**。
直前版の5関数（stateless uncached candidateを含む）の**SASS instruction bodyが完全一致**。
[device audit](bvh_normal_device_code_20261012.json)に全function hashとresourceを保存した。

baselineは直前のCPU geometry/scratch cacheのsource5件と実pydへSHA固定。
GPU順次・5交互round×20、全**48 timed条件**のraw samples/round中央値/全timed出力hashを
[測定JSON](bvh_normal_gpu_20261012.json)に保存。
**110 LOS/反射の検証ケース**でLOS・反射点・遅延がビット一致。
random/mesh tails/ray境界/ECEF/tie/reversed root/nonfinite/degenerate/深い木/重複leaf/
underflow・大座標/normal cutoff/reuse/owned snapshotを含む。
**459関連テストPASS**、新43件のmemcheck/racecheck/synccheckもすべて0（leak0）。
PF/signalの直前sourceとbuild/installed binaryはSHA一致で保持した。

[再現patch](probes/bvh_resident_normal_cache_20261012.patch)は直前測定sourceから今回の
CRLF sourceをbyte単位で復元できることを確認した。
[benchmark runner](bench_bvh_normal_compare.py)、[lifetime/精度テスト](../tests/test_bvh_normal_cache.py)。
実ビルド比較の再現コマンド:

```powershell
python benchmarks/bench_bvh_normal_compare.py --baseline dist/bvh_normal_baseline/_bvh.cp312-win_amd64.pyd --candidate build/diag/Release/_bvh.cp312-win_amd64.pyd --baseline-dir dist/bvh_normal_baseline --rounds 5 --repeats 20 --output dist/bvh_normal_final_20261012.json
```

## Signal global-descriptor coefficient cache (2026-10-12)

搬送波係数のGPU block共有を、global descriptorを使う**33〜256チャンネル**へ広げた。
公開256ch/2,600,000 samplesは **77.9126→61.3208 ms
（1.271×）（約21.3%短縮）**。
別の5交互round×20による追加確認でも **76.9202→59.0877 ms
（1.302×）** となった。対象は多チャンネルの長い信号生成。
一律の都市シミュレーション全体の利得は確認できていない。

### 同期・転送・返却配列を含む信号生成

| 条件 | 旧 (ms) | 新 (ms) | 速度比 |
|:---|---:|---:|---:|
| native / GPS33 / 260,000 samples | 2.3631 | 1.9249 | 1.228× |
| native / GPS65 / 260,000 samples | 5.9867 | 4.9687 | 1.205× |
| native / GPS128 / 260,000 samples | 9.5434 | 7.6532 | 1.247× |
| native / GPS256 / 260,000 samples | 9.1482 | 7.8000 | 1.173× |
| native / mixed nav256 / 260,000 samples | 10.8722 | 8.9108 | 1.220× |
| public / GPS65 / 260,000 samples | 9.4215 | 7.8415 | 1.201× |
| public / GPS128 / 2,600,000 samples | 41.6434 | 34.3307 | 1.213× |
| public / GPS256 / 2,600,000 samples | 77.9126 | 61.3208 | 1.271× |

### 短いepoch・未変更経路の対照

| 条件 | 旧 (ms) | 新 (ms) | 速度比 |
|:---|---:|---:|---:|
| native / GPS65 / 2,600 samples | 0.2560 | 0.2434 | 1.052× |
| public / GPS65 / 2,600 samples | 0.6081 | 0.6903 | 0.881× |
| native / unchanged inline32 / 260,000 samples | 4.5476 | 4.6787 | 0.972× |
| native / unchanged noise-only / 260,000 samples | 0.9743 | 1.2808 | 0.761× |
| public / unchanged fallback257 / 260,000 samples | 12.5670 | 10.1734 | 1.235× |

短いepochに一律の利得はない。未変更257chの本測定にも大きな変動があり、負荷の影響を
区別するためGLO65/public65/fallback257/長い256ch/Urban全体の5条件を追加確認した。
fallback257は追試で **1.006×**、
GLO65は本測定 **0.889×** から追試
**1.224×** へ変わった。
本測定の値を置き換えず、[追加確認](signal_global_frequency_check_20261012.json) に全rawを保存する。

### 32衛星の都市シミュレーション全体

| 三角形 / samples / 実生成ch | 旧 (ms) | 新 (ms) | 速度比 |
|:---|---:|---:|---:|
| 1,200 / 2,600 / 46 | 1.0532 | 0.8634 | 1.220× |
| 1,200 / 260,000 / 46 | 3.1267 | 2.6562 | 1.177× |
| 12,000 / 2,600 / 53 | 3.3772 | 3.3774 | 1.000× |
| 12,000 / 260,000 / 53 | 8.0341 | 7.2775 | 1.104× |

LOS/反射/channel構築/実CUDA IQ/同期・ホスト返却を含み、CPU BVH構築は測定外。
12,000三角形/260,000 samplesの全体は本測定で **1.104×** だが、追試の
median-of-rounds比較では **4.4096→5.1014 ms
（0.864×）** に逆転した。追試のround対では5回中
4回で新版が速く、途中で旧新版とも時間帯が変わっている。
host/GPU非専有の全体wall timeから固定利得を主張せず、両結果を掲載する。
信号生成で確認した速度比と過去のBVH/inline版の速度比は掛け合わせない。

GPU kernelの最初のn_channels threadが、元のFP64式
`kTwoPi * (intermediate_freq + doppler_hz)` をblockごとに一度計算する。
256 threads/blockに収まる上限をcompile-time assertで守り、全threadがbarrierを通った後に
partial blockの範囲外threadがreturn。CPU係数計算/近似逆数/演算順変更を導入していない。
時間division/chip式/整数剰余/phase最終演算/sincos/nav clamping/FP32累積順/Philox/noiseは同じ。
CPU packing、descriptor/navのH2D、出力D2H、launch数/block256/API/owner allocation量も同じ。
追加kernelは**register42（旧global44）/static shared2048 bytes/stack・local0**。
既存inline/noise-only/single/uncached-globalの**4関数SASSが全一致**。
1〜32chは直前のinline cache、257ch以上は元のglobal経路を保つ。
[device audit](signal_global_frequency_device_code_20261012.json) にresource/hashと
[source patch](probes/signal_global_frequency_cache_20261012.patch) を保存した。
PF/直前BVHのsource/build/installed binary hashは同一。

baselineの5 source/binary hashは [直前inline係数版](signal_frequency_gpu_20261012.json) に固定。
CUDA12.8/MSVC Release/compute capability7.5/C++20/fast mathなし。global型登録を分離した
旧新版のpersistent processを使い、GPU処理は常に順次、**stageごと5交互round×20回**。
参照出力1回/各3 warm-up、同期込みwall time、表はround中央値の中央値。
IPC/hash/CUDA・code-table初期化/CPU mesh・BVH構築は測定外。
全48条件のraw/round中央値/source・binary・runner hashesは
[本測定JSON](signal_global_frequency_gpu_20261012.json)。
public256ch/2.6m samplesのround範囲（本測定）: baseline 77.5366–79.1959 ms / candidate 59.6611–87.5688 ms。
一度新版が遅いroundもあり、追加確認の範囲は baseline 76.7034–76.9980 ms / candidate 59.0109–59.6506 ms。
非専有hostの変動、小条件の低下、全体の追試逆転を含めて公開する。

**457条件のIQ bit hashes一致**。各build内でもwarm/coldを全uint32比較し、全timed IQ/Urban
metadata hash一致。33/64/65/127/128/255/256/257/513、partial blocks、mixed nav、全5星座、
整数端/FP64 cancellation、無効channel filtering、grow/shrink/入力更新/返却配列を含む。
関連 **416テストPASS**。新55件は変更していないsingle/257ch以上のkernelとのbit比較、
境界/同じ入力storage更新を検査する。既存inline参照テストのpaddingも257chへ移し、
引き続き元のuncached式を独立参照する。両test file計104件のCUDA
**memcheck0 errors/0 leaked、racecheck0 hazards/errors/warnings、synccheck0 errors**。

再現（baselineは上記5 source/binary hashに一致する保存版）:

```powershell
$env:PYTHONPATH="python"
python benchmarks/bench_signal_global_frequency_compare.py --baseline <baseline-pyd> --candidate <candidate-pyd> --baseline-dir <baseline-source-dir> --rounds 5 --repeats 20 --output <output-json>
python -m pytest tests/test_signal_global_frequency_cache.py tests/test_signal_frequency_cache.py -q
```

## Signal frequency coefficient cache (2026-10-12)

32チャンネル以下のfused IQ kernelで、チャンネルごとの搬送波係数
`kTwoPi * (intermediate_freq + doppler_hz)` をGPU上でblockごとに一度計算し、
共有メモリから参照する。元のFP64加算/乗算を維持して、サンプルごとの重複演算を減らした。
公開12チャンネル/2,600,000サンプルは **24.3171→19.8292 ms
（1.226×）**、12,000三角形/12衛星/260,000サンプルのwarm都市シミュレーション全体は
**5.8920→5.4444 ms（1.082×）**。
後者はLOS/反射/channel metadata/実CUDA IQ/ホスト出力まで含む。CPU BVH構築は測定外で、
直前のBVH構築速度比とは掛け合わせていない。

### 信号生成（同期・転送・返却配列を含む）

| 条件 | 旧 (ms) | 新 (ms) | 速度比 |
|:---|---:|---:|---:|
| native / GPS12 / 260,000 samples | 1.1677 | 1.0667 | 1.095× |
| native / GPS32 / 260,000 samples | 4.2044 | 3.4290 | 1.226× |
| native / GPS12 / 2,600,000 samples | 24.4132 | 19.8759 | 1.228× |
| native / GPS32 / 2,600,000 samples | 42.3473 | 35.6311 | 1.188× |
| native / mixed nav32 / 260,000 samples | 4.8563 | 4.1265 | 1.177× |
| public / GPS12 / 260,000 samples | 2.8260 | 2.5105 | 1.126× |
| public / GPS12 / 2,600,000 samples | 24.3171 | 19.8292 | 1.226× |
| cold standalone / GPS12 / 260,000 samples | 3.9343 | 3.5939 | 1.095× |

### 都市シミュレーション全体（warm BVH、12衛星）

| 三角形 / samples | 旧 (ms) | 新 (ms) | 速度比 |
|:---|---:|---:|---:|
| 12 / 2,600 | 0.7053 | 0.6864 | 1.028× |
| 12 / 260,000 | 2.3012 | 2.3323 | 0.987× |
| 1,200 / 2,600 | 0.7363 | 1.0629 | 0.693× |
| 1,200 / 260,000 | 3.7778 | 3.4454 | 1.096× |
| 12,000 / 2,600 | 3.0541 | 3.1381 | 0.973× |
| 12,000 / 260,000 | 5.8920 | 5.4444 | 1.082× |

### 未変更kernel/短いepochの対照

| 条件 | 旧 (ms) | 新 (ms) | 速度比 |
|:---|---:|---:|---:|
| native / noise only / 2,600,000 samples | 10.0264 | 9.9995 | 1.003× |
| native / GPS33 / 260,000 samples | 4.6815 | 4.6468 | 1.007× |
| native / GPS65 / 260,000 samples | 4.8246 | 4.8223 | 1.000× |
| native / GPS12 / 2,600 samples | 0.0779 | 0.0751 | 1.037× |
| native / GPS32 / 2,600 samples | 0.1401 | 0.1436 | 0.976× |

短いepochに一律の利得はない。1,200三角形/2,600サンプルは初回表で0.693×となったが、
roundが約0.7/1.1 msに分かれ、旧新版のmedian-of-roundsが異なる側になった。
懸念を確認するため同条件/単独native/長いepochのみ **5交互round×20** で再測定し、
短い都市epochは **0.4723→0.4847 ms（0.975×）**、
同じmeshの260,000サンプルは **1.6836→1.5749 ms（1.069×）**。
初回値を置き換えず、[追加確認](signal_frequency_short_check_20261012.json) に全rawを保存した。
短い条件の小幅低下/非専有hostの変動を含めて記録し、対象を多数サンプルの32チャンネル以下とする。

初めのn_channels threadがGPU上で係数を作り、全blockが `__syncthreads` に参加した後に
partial blockの範囲外threadがreturnする。`t=idx/sampling_freq`、chip indexのFP64式/整数剰余、
phaseの最終乗算・加算、`__sincosf`、nav-bit clamping、入力順のFP32積/加算、Philoxの
seed/sequence/乱数の順序、noise式を保持。CPU係数計算/近似逆数/演算順変更は導入していない。
block256/launch数/API/CPU packing/転送数/owner buffer保持量も同じ。
inline kernelはregister32のまま、stack/local0、static shared **0→256 bytes**、block同期を1回追加。
33チャンネル以上のglobal descriptor/noise-only/single kernelは全SASS一致で、
register44/50/26、stack/local/shared0のまま。[device audit](signal_frequency_device_code_20261012.json)
にresource/hashと [source patch](probes/signal_frequency_cache_20261012.patch) を保存した。
PF/直前BVHのsourceとbuild/installed binary hashは同一。

baselineの5 source/binary hashは [signal workspace版](signal_workspace_gpu_20261012.json) に固定。
CUDA12.8/MSVC Release/compute capability7.5/C++20/fast mathなし。
pybind workspaceのglobal型登録を分離するため、旧新版の実production extensionを別processへ読み込む。
2つのpersistent workerを用い、GPU処理は常に順次実行。**stageごと5round×20回、旧新版順を交互**。
初期の出力参照1回/各3 warm-up、同期込みwall time、表はround中央値の中央値。
IPC/出力hash/CUDA・code-table初期化/CPU mesh・BVH構築は測定外。
host/GPUは非専有。[本測定JSON](signal_frequency_gpu_20261012.json) は全54条件のraw/round中央値/
source/binary/runner hashesを含む。公開12ch/2,600,000 samplesのround範囲:
baseline 24.2694–24.5010 ms / candidate 19.7950–20.2931 ms。
12,000三角形/260,000 samplesのUrban全体:
baseline 5.8171–5.9488 ms / candidate 5.4140–5.8004 ms。

**377条件のIQ bit hashes一致**。各build内でもwarm/cold IQを全uint32比較し、全測定呼出しの
返却IQ/Urban metadata hashが一致。全5星座、負位相/整数端、chip/nav境界、32/33境界、
mixed nav、seed最大値、sum順、underflow、所有/入力不変、grow/shrinkを含む。
関連 **361テストPASS**、追加49件は変更していないsingle/global kernelとのbit比較、
FP64係数のcancellation/大値/極小値、partial block、交互dispatch/返却配列を検査する。
49件のCUDA **memcheck0 errors/0 leaked、racecheck0 hazards/errors/warnings、synccheck0 errors**。

先に試した固定code lengthの整数剰余案は377条件でbit一致したが、3交互round×10の
大条件/全体に安定した利得がなく、採用せず元の剰余を保持した。
[棄却測定](signal_modulo_rejected_20261012.json) と
[復元patch](probes/signal_modulo_rejected_20261012.patch) を保存した。
非交互の初期1roundや予備の大きな速度比を採用根拠にはしない。

再現（baselineは上記5 source/binary hashに一致する保存版）:

```powershell
$env:PYTHONPATH="python"
python benchmarks/bench_signal_kernel_compare.py --baseline <baseline-pyd> --candidate <candidate-pyd> --baseline-dir <baseline-source-dir> --rounds 5 --repeats 20 --output <output-json>
python -m pytest tests/test_signal_frequency_cache.py -q
```

## BVH construction geometry/scratch reuse (2026-10-12)

シーンを新しく作る/三角形を更新する際のCPU SAH構築を高速化した。三角形ごとの
境界箱と3軸の重心を一度だけ計算し、再帰処理/ビン集計/ソートで参照する。
prefix/suffix境界箱の作業領域も構築ごとに一度確保する。親は分割位置を確定した後に
作業領域を使わないため、子の再帰処理で同じ領域を上書きできる。
12,000三角形/12衛星/260,000サンプルの**シーン構築→IQ出力→close全体**は
**48.8338→38.0132 ms（1.285×）**、
120,000三角形では **479.2411→341.4804 ms（1.403×）**。
対象は初期化/メッシュ更新。既に構築したシーンを使うepochの高速化は主張しない。

### CPU BVH構築（native bindingの入力検査/返却配列を含む）

| 三角形数 | 旧 (ms) | 新 (ms) | 速度比 |
|---:|---:|---:|---:|
| 12 | 0.0090 | 0.0079 | 1.139× |
| 1,200 | 1.2228 | 0.7891 | 1.550× |
| 12,000 | 19.5826 | 12.5678 | 1.558× |
| 120,000 | 383.5301 | 245.1621 | 1.564× |

### 公開UrbanSignalSimulatorの構築→epoch→GPU領域解放

12衛星/260,000サンプル。Pythonのmesh snapshot/並べ替え、CPU BVH、GPU転送/確保、
LOS/反射、channel metadata、実CUDA IQ、同期/ホスト出力/closeまでを含む。

| 三角形数 | 旧 (ms) | 新 (ms) | 速度比 |
|---:|---:|---:|---:|
| 12 | 2.2369 | 2.2325 | 1.002× |
| 1,200 | 6.8806 | 6.2100 | 1.108× |
| 12,000 | 48.8338 | 38.0132 | 1.285× |
| 120,000 | 479.2411 | 341.4804 | 1.403× |

### 既存simulatorのupdate_triangles→epoch

信号workspaceを保持したままメッシュを再構築し、シーンを再uploadしてIQを生成する。

| 三角形数 | 旧 (ms) | 新 (ms) | 速度比 |
|---:|---:|---:|---:|
| 12 | 1.6771 | 1.6929 | 0.991× |
| 1,200 | 6.0291 | 5.5655 | 1.083× |
| 12,000 | 42.0948 | 33.4113 | 1.260× |
| 120,000 | 387.3620 | 274.5549 | 1.411× |

### メッシュを更新しないwarm epoch（対照）

| 三角形数 | 旧 (ms) | 新 (ms) | 速度比 |
|---:|---:|---:|---:|
| 12 | 1.5339 | 1.5575 | 0.985× |
| 1,200 | 4.9290 | 4.7753 | 1.032× |
| 12,000 | 2.5991 | 2.5793 | 1.008× |
| 120,000 | 5.2953 | 5.2854 | 1.002× |

元のFP64の境界箱/重心式、和の順序、SAHの3軸/12bin/厳密比較、sortの比較式/同値の扱い、
prefix/suffixの累積順、leaf上限4、分割位置/ノード順/三角形順を保持した。
構築完了後はcache/scratchを破棄し、mesh更新ごとに計算し直す。API/常駐GPU buffer/転送数は同じ。
構築用geometryは72*n_tri bytes、prefix/suffixは各48*n_tri bytes（n_tri>4）で、一時領域合計
168*n_tri bytes（120,000三角形で20.16 MB）。元の各再帰nodeの2作業vectorを置き換える値で、
プロセス全体のpeakメモリを測った数値ではない。小さいleafではprefix/suffixを確保しない。
GPU sectionのsource bytesと全5 device関数のSASSが一致した。独立の
[device code audit](bvh_build_device_code_20261012.json) にhashを保存。
変更を復元する [source patch](probes/bvh_build_geometry_scratch_20261012.patch) も添付した。
PF/信号generatorのsourceとbuild/installed binaryも直前測定版と同一。

[直前combined BVH](bvh_combined_gpu_20261011.json) の5 source/binary hashをbaselineとして固定。
CUDA12.8/MSVC Release/compute capability7.5/C++20/fast mathなし。
同一processで実旧新版を順次実行し、**5round×20回、呼出し順を交互**にした。
各3 warm-up、同期込みwall time、表はround中央値の中央値。mesh生成、CUDA/code-table初期化、
出力hashは測定外。公開wrapperのbackend選択は両側に同じ処理を含める。
host/GPUは非専有で、stage間/round間の負荷変動がある。全28条件のraw samples/round中央値/
source/binary/runner hashesは [JSON](bvh_build_gpu_20261012.json)。
12,000三角形のnative構築のround範囲: baseline 19.3826–24.8668 ms / candidate 12.2258–14.7848 ms。
同条件のUrban構築+epochのround範囲: baseline 41.9346–49.1255 ms / candidate 36.7967–38.7875 ms。
120,000三角形Urban全体ではroundごとの速度比は1.303–1.529×。
round間のhost負荷変動により旧新版のround範囲は重なるが、各roundでは新版が速かった。
小mesh/warm controlの変動も掲載し、全条件一定の利得や過去の速度比の積は主張しない。

**97構築条件のノード/三角形順が全bit一致**。うち87条件で独立に構築した両GPU sceneの
LOS/反射点/遅延も全bit一致。入力不変、返却配列の所有、同じstorageの値変更、
leaf境界、ランダム/同重心/同一三角形/点/平面/直線/signed zero/ECEF/非連続/float32/int/
極小/極大有限値を検証した。全28測定条件のすべての返却値にもhash一致を要求し、
UrbanはIQとchannel metadataを含む。関連 **312テストPASS**。
追加44件はtreeの独立なcoverage/bounds検査と再構築後のlinear ray参照を含み、
CUDA memcheck **0 errors / 0 bytes leaked**。
完全退化meshは257三角形までを比較した。旧版は1,025三角形の直線meshで既存の
再帰stack上限に達するため、この大きな退化条件を性能/一致保証に含めていない。
cache化によるstack上限の解消はこの変更の対象外。

再現（旧版は上記hashと一致するsourceからbuildし、5 sourceをbasenameでbaseline-dirへ保存）:

```powershell
$env:PYTHONPATH="python"
python benchmarks/bench_bvh_build_compare.py --baseline <baseline-pyd> --candidate <candidate-pyd> --baseline-dir <baseline-source-dir> --rounds 5 --repeats 20 --output <output-json>
python -m pytest tests/test_bvh_build_cache.py -q
```

## Student-t PR specialization (2026-10-12)

nu>0かつparticle gateが無効なStudent-t尤度を専用kernelに分け、逆スケールと係数をGPU上で
blockごとに一度計算する。100万粒子/12衛星/nu3の単独weightは
**8.4165→7.9962 ms（1.053×）**、公開predict/Student PR/ESS/条件付きB60 resample/estimate
全体は **17.1683→16.7782 ms（1.023×）（約2.3%短縮）**。
4衛星の単独は **3.0476→2.7484 ms（1.109×）**、32衛星は
**21.8407→21.1264 ms（1.034×）**。Student PR+Doppler KF全体は
**46.5072→46.1208 ms（1.008×）** と小さい。直前KF-launch版との実測差で、過去の速度比を
掛け合わせていない。対象はStudent-tを指定する経路。

元のGPU式 `1.0/(nu*sigma_pr*sigma_pr)` と `0.5*(nu+1.0)`、range/residual、
`-half_nup1*log(1.0 + ws*residual*residual*inv_scale)` と衛星の累積順/FP64を維持し、loopを2件unroll。
CPU逆数/近似log/観測間の和の並べ替えを導入していない。Student優先を保つため、Huber指定時も
nu>0/gate無効なら専用kernelを使い、正のgateは元のgeneralへdispatchする。nu<=0は既存の
Gaussian/Huber/gate経路を維持。zero weightをskipせず、underflow/overflow時の元のNaN動作も検証した。
log weightは元と同じoverwrite。3 H2Dコピー/stream同期/block256/launch数/APIは同じ。
追加owner buffer/cudaMallocは0。新kernel register48（旧general44）、stack/local0、static shared16 bytes。
dynamic sharedは5*n_sat*8 bytesで同じ。1,228衛星でも49,120+16=49,136 bytesとなり48 KiB以内。

baselineの5 source/binary hashを [直前KF-launch版](pf_kf_launch_gpu_20261012.json) のcandidateに固定。
CUDA12.8/MSVC Release/compute capability7.5/C++20/fast mathなし。
別processの実production extensionをGPUで順次実行し、**5round×20回、旧新版順を交互**にした。
3 warm-up後の同期込みwall time、表はround中央値の中央値。初期化/入力復元/hash/state dumpは測定外。
host/GPUは非専有。単独weightもbinding/3 H2D/kernel/最終syncを含む。全80条件とraw samples、
source/binary/benchmark hashesは [JSON](pf_student_gpu_20261012.json)。

### nu3の単独Student weight

| 粒子数 | 処理 | 旧 (ms) | 新 (ms) | 速度比 |
|---:|:---|---:|---:|---:|
| 2,048 | Student_sat4 | 0.0786 | 0.0624 | 1.261× |
| 2,048 | Student_sat12 | 0.0966 | 0.0967 | 0.999× |
| 2,048 | Student_sat16 | 0.1195 | 0.1114 | 1.074× |
| 2,048 | Student_sat32 | 0.1798 | 0.1729 | 1.040× |
| 2,048 | Student_sat64 | 0.3174 | 0.3080 | 1.031× |
| 2,048 | Student_sat65 | 0.3301 | 0.3054 | 1.081× |
| 10,000 | Student_sat4 | 0.0883 | 0.0925 | 0.955× |
| 10,000 | Student_sat12 | 0.1497 | 0.1426 | 1.050× |
| 10,000 | Student_sat16 | 0.1934 | 0.1870 | 1.034× |
| 10,000 | Student_sat32 | 0.3278 | 0.3020 | 1.085× |
| 10,000 | Student_sat64 | 0.5956 | 0.5619 | 1.060× |
| 10,000 | Student_sat65 | 0.6005 | 0.5734 | 1.047× |
| 100,000 | Student_sat4 | 0.3498 | 0.3308 | 1.057× |
| 100,000 | Student_sat12 | 0.9672 | 0.8821 | 1.097× |
| 100,000 | Student_sat16 | 1.2200 | 1.1519 | 1.059× |
| 100,000 | Student_sat32 | 2.3838 | 2.2786 | 1.046× |
| 100,000 | Student_sat64 | 4.6123 | 4.4536 | 1.036× |
| 100,000 | Student_sat65 | 4.6828 | 4.5259 | 1.035× |
| 1,000,000 | Student_sat4 | 3.0476 | 2.7484 | 1.109× |
| 1,000,000 | Student_sat12 | 8.4165 | 7.9962 | 1.053× |
| 1,000,000 | Student_sat16 | 11.1136 | 10.6212 | 1.046× |
| 1,000,000 | Student_sat32 | 21.8407 | 21.1264 | 1.034× |
| 1,000,000 | Student_sat64 | 43.2874 | 42.1116 | 1.028× |
| 1,000,000 | Student_sat65 | 43.9471 | 42.7725 | 1.027× |

nu1/nu10の12衛星呼出しも全粒子数についてJSONに含む。

### 12衛星Student更新全体

native/publicはpredict→Student PR(nu3)→任意のDoppler KF→ESS→条件付きB60 resample→estimateまで。
公開wrapperの入力処理と転送も含む。high_ESSはsigma_pr1000でresampleを回避。

| 粒子数 | 処理 | 旧 (ms) | 新 (ms) | 速度比 |
|---:|:---|---:|---:|---:|
| 2,048 | native_PR_step | 0.2500 | 0.2070 | 1.208× |
| 2,048 | native_PR_KF_step | 0.3515 | 0.3306 | 1.063× |
| 2,048 | native_PR_high_ESS_step | 0.2098 | 0.1866 | 1.125× |
| 2,048 | public_PR_step | 0.2736 | 0.2611 | 1.048× |
| 2,048 | public_PR_KF_step | 0.3837 | 0.3864 | 0.993× |
| 10,000 | native_PR_step | 0.3427 | 0.3042 | 1.127× |
| 10,000 | native_PR_KF_step | 0.7016 | 0.6792 | 1.033× |
| 10,000 | native_PR_high_ESS_step | 0.3049 | 0.2912 | 1.047× |
| 10,000 | public_PR_step | 0.3165 | 0.3648 | 0.868× |
| 10,000 | public_PR_KF_step | 0.7273 | 0.7119 | 1.022× |
| 100,000 | native_PR_step | 1.6475 | 1.5749 | 1.046× |
| 100,000 | native_PR_KF_step | 4.6592 | 4.6062 | 1.012× |
| 100,000 | native_PR_high_ESS_step | 1.3921 | 1.3314 | 1.046× |
| 100,000 | public_PR_step | 1.7360 | 1.6177 | 1.073× |
| 100,000 | public_PR_KF_step | 4.7496 | 4.6390 | 1.024× |
| 1,000,000 | native_PR_step | 17.1902 | 16.7521 | 1.026× |
| 1,000,000 | native_PR_KF_step | 46.4957 | 46.0954 | 1.009× |
| 1,000,000 | native_PR_high_ESS_step | 11.6092 | 11.1602 | 1.040× |
| 1,000,000 | public_PR_step | 17.1683 | 16.7782 | 1.023× |
| 1,000,000 | public_PR_KF_step | 46.5072 | 46.1208 | 1.008× |

### 未変更control

| 粒子数 | 処理 | 旧 (ms) | 新 (ms) | 速度比 |
|---:|:---|---:|---:|---:|
| 2,048 | Gaussian_control | 0.0666 | 0.0643 | 1.035× |
| 2,048 | Huber_control | 0.0908 | 0.0802 | 1.131× |
| 2,048 | Gate_control | 0.0919 | 0.0795 | 1.156× |
| 2,048 | Student_gate_control | 0.1242 | 0.1106 | 1.123× |
| 2,048 | predict_control | 0.0277 | 0.0252 | 1.097× |
| 2,048 | resample_control | 0.0334 | 0.0308 | 1.084× |
| 2,048 | KF_control | 0.1253 | 0.1159 | 1.080× |
| 10,000 | Gaussian_control | 0.0970 | 0.0822 | 1.180× |
| 10,000 | Huber_control | 0.1159 | 0.1146 | 1.011× |
| 10,000 | Gate_control | 0.1245 | 0.1195 | 1.041× |
| 10,000 | Student_gate_control | 0.1951 | 0.1825 | 1.069× |
| 10,000 | predict_control | 0.0385 | 0.0383 | 1.005× |
| 10,000 | resample_control | 0.0382 | 0.0375 | 1.019× |
| 10,000 | KF_control | 0.3755 | 0.3624 | 1.036× |
| 100,000 | Gaussian_control | 0.3703 | 0.3721 | 0.995× |
| 100,000 | Huber_control | 0.6442 | 0.6361 | 1.013× |
| 100,000 | Gate_control | 0.6931 | 0.6805 | 1.018× |
| 100,000 | Student_gate_control | 1.2378 | 1.1970 | 1.034× |
| 100,000 | predict_control | 0.2350 | 0.2336 | 1.006× |
| 100,000 | resample_control | 0.2256 | 0.2329 | 0.968× |
| 100,000 | KF_control | 2.8170 | 2.8144 | 1.001× |
| 1,000,000 | Gaussian_control | 3.2447 | 3.2360 | 1.003× |
| 1,000,000 | Huber_control | 5.7764 | 5.7793 | 0.999× |
| 1,000,000 | Gate_control | 6.2161 | 6.2038 | 1.002× |
| 1,000,000 | Student_gate_control | 11.0727 | 11.0506 | 1.002× |
| 1,000,000 | predict_control | 2.1456 | 2.1383 | 1.003× |
| 1,000,000 | resample_control | 2.1048 | 2.0957 | 1.004× |
| 1,000,000 | KF_control | 26.9100 | 26.9002 | 1.000× |

100万粒子の単独Student12のround中央値はbaseline8.2904〜8.4788 ms、candidate7.9411〜8.0124 ms、公開Student全体は
baseline17.0692〜17.2398 ms、candidate16.6940〜16.8020 msで双方の範囲が分離した。100万粒子の未変更predict/resample/KFは
1.003/1.004/
1.000×。小粒子数のhost変動/悪化も表とraw sampleに残す。
予備測定の大きな速度比を代表値にしない。全条件の固定利得、実データ位置精度、whole GNSS・IMUの
利得は未評価。既存systematic Thrust CDF境界変動は範囲外。

**2,185ケースで全16状態/log weight/更新軌跡がbit一致**。1〜100万粒子、1〜128衛星、nu0/負nu/
nu0.5/1/2/3/30/微小・巨大nu、sigma² underflow/overflow、非有限状態、zero weight/near range、
Huber優先順位/gateとfallback/buffer growth、12件の5epoch Student PR/PR+KF/high_ESS軌跡を含む。
全80測定条件/全round/各回の返却値hashと各stage最後のstate/log-weight snapshotも一致した。
時間測定中の全回でstate dumpは行っていない。

関連PF/wrapper/smoother/SMC/IMU **647テストPASS**。追加77件は独立NumPy Student参照、元general
とのbit比較、Student/Huber優先、overwrite/返却array所有権、極端scaleとzero weight、partial block、
1,228衛星shared/growth境界を検証。CUDA **memcheck0 errors/0 bytes leaked、racecheck0 hazards/errors/
warnings、synccheck0 errors**。

先にKFのloop展開抑制/固定12衛星/分母共有を試したが採用しなかった。各1,728ケースでbit一致したが、
大粒子数の利得が乏しい、または遅くなった。分母共有はCUDAのRN reciprocalとRD/RU FMAから求める
区間のRN端点が一致した場合に使い、曖昧なら元のdivisionへ戻す設計だった。
使用した丸め契約は [CUDA12.8 double intrinsics](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html)。
PFのbit一致は検証したが、独立の全域division検査を実施したとはしない。KF12は予備測定で
26.5951→28.4736 msと約7%遅かった。3案は**非交互1round×20回の予備測定**で採用証拠にせず、
raw timings/検証manifest hash/source・binary hashと復元patch/追加headerを
[不採用probe JSON](pf_kf_math_rejected_20261012.json) に保存した。元の128-thread KFは維持した。

```powershell
cmake --build build/diag --config Release --target _gnss_gpu_pf_device --parallel 4
$env:PYTHONPATH='python'
python benchmarks/bench_pf_student_compare.py --baseline dist/pf_kf_loop_baseline/_gnss_gpu_pf_device.cp312-win_amd64.pyd --candidate build/diag/Release/_gnss_gpu_pf_device.cp312-win_amd64.pyd --baseline-dir dist/pf_kf_loop_baseline --rounds 5 --repeats 20 --output dist/pf_student_final_20261012.json
python -m pytest tests/test_pf_student_weight.py -q
```

baseline_dirには直前KF-launch版の5 sourceのbasenameとbinaryをbyte-exact保存したものを指定する。
比較は [script](bench_pf_student_compare.py)、追加検証は [test](../tests/test_pf_student_weight.py)。

## PF Doppler KF launch layout (2026-10-12)

32衛星以下のDoppler KFを256→128 threads/blockに変更した。2,048粒子/12衛星の単独native呼出しは
**0.2009→0.1173 ms（1.713×）**、32衛星は **0.4775→0.2568 ms（1.859×）**。
12衛星のpredict/PR/KF/ESS/条件付きB60 resample/estimate全体は、nativeで
**0.4205→0.3266 ms（1.288×）**、公開Python APIで
**0.4565→0.3586 ms（1.273×）**（約21.4%短縮）。少数粒子に効く変更で、
100万粒子/12衛星の公開全体は **41.6011→41.6155 ms（1.000×）** とほぼ差がない。
直前の予測専用化版との実測比較で、過去の速度比を掛け合わせていない。

変更はhostのblock/grid指定だけ。2,048粒子では8→16 blocksに分け、32衛星超のfallbackは従来の
256 threads/blockを維持する。他のquery/resamplerに使うstate.grid_sizeも従来の256-thread基準のまま。
全device kernelの `cuobjdump --dump-sass` 出力が旧新版でbyte-identicalだった（このCUDA12.8 build）。
FP64式/除算/sqrt/log/FMA/衛星累積順、geometry cache、KF covariance/clamp/尤度、乱数を変えていない。
SASS出力のLF-normalized UTF-8 SHA256は両方
`ddf68d2003bb176a6b6a9c69cfdf1c6871dba44ccabbfe1540529fa4b5b4f9d1`。
registerはcache16/cache32が86/86、fallback94。stack512/1024/0 bytes、static shared/local0で同じ。
blockごとのdynamic sharedも同じ（64衛星以下9*n_sat*8、超過時8*n_sat*8 bytes）。
追加owner buffer/cudaMalloc/転送/launch/syncは0。実occupancyやpeak memoryは測定していない。

baselineの5 source/binary hashを [直前予測版](pf_predict_gpu_20261012.json) のcandidateへ固定。
CUDA12.8/MSVC Release/compute capability7.5/C++20/fast mathなし。
別processの実production extensionをGPU上で順次実行し、**5round×20回、旧新版順を交互**にした。
3 warm-up後の同期込みwall time、表はround中央値の中央値。初期化/入力復元/hash/state dumpは測定外。
単独KFもbinding/observation packing/H2D/kernel/最終syncを含む。host/GPUは非専有。
全72条件とraw samples/source/binary/benchmark hashesは [JSON](pf_kf_launch_gpu_20261012.json)。

### KF単独呼出し

| 粒子数 | 処理 | 旧 (ms) | 新 (ms) | 速度比 |
|---:|:---|---:|---:|---:|
| 2,048 | KF_sat4 | 0.0958 | 0.0616 | 1.555× |
| 2,048 | KF_sat12 | 0.2009 | 0.1173 | 1.713× |
| 2,048 | KF_sat16 | 0.2576 | 0.1488 | 1.730× |
| 2,048 | KF_sat17 | 0.2710 | 0.1587 | 1.708× |
| 2,048 | KF_sat32 | 0.4775 | 0.2568 | 1.859× |
| 2,048 | KF_sat33 | 0.5974 | 0.5959 | 1.002× |
| 2,048 | KF_sat65 | 1.1761 | 1.1754 | 1.001× |
| 10,000 | KF_sat4 | 0.1515 | 0.1522 | 0.995× |
| 10,000 | KF_sat12 | 0.3672 | 0.3594 | 1.022× |
| 10,000 | KF_sat16 | 0.4696 | 0.4678 | 1.004× |
| 10,000 | KF_sat17 | 0.5029 | 0.4995 | 1.007× |
| 10,000 | KF_sat32 | 0.9019 | 0.9027 | 0.999× |
| 10,000 | KF_sat33 | 1.1482 | 1.1494 | 0.999× |
| 10,000 | KF_sat65 | 2.3511 | 2.3564 | 0.998× |
| 100,000 | KF_sat4 | 1.0062 | 0.9850 | 1.022× |
| 100,000 | KF_sat12 | 2.9194 | 2.8405 | 1.028× |
| 100,000 | KF_sat16 | 3.8170 | 3.7171 | 1.027× |
| 100,000 | KF_sat17 | 4.0428 | 3.9379 | 1.027× |
| 100,000 | KF_sat32 | 7.4726 | 7.2690 | 1.028× |
| 100,000 | KF_sat33 | 9.5563 | 9.5464 | 1.001× |
| 100,000 | KF_sat65 | 19.3470 | 19.3134 | 1.002× |
| 1,000,000 | KF_sat4 | 9.3626 | 9.3651 | 1.000× |
| 1,000,000 | KF_sat12 | 26.9121 | 26.9158 | 1.000× |
| 1,000,000 | KF_sat16 | 35.6780 | 35.6872 | 1.000× |
| 1,000,000 | KF_sat17 | 37.8512 | 37.8580 | 1.000× |
| 1,000,000 | KF_sat32 | 70.7367 | 71.3319 | 0.992× |
| 1,000,000 | KF_sat33 | 90.6339 | 90.6859 | 0.999× |
| 1,000,000 | KF_sat65 | 184.2665 | 184.2622 | 1.000× |

粒子ごとに異なる相関SPD共分散を復元するKF_imported_sat12/32、同一ownerで5回連続更新する
KF_five_updatesもJSONに含む。import/復元は測定外、連続5回のpacking/転送/同期は測定内。

### 12衛星の更新全体

native/publicともpredict→PR→Doppler KFの両尤度→ESS→条件付きB60 resample→estimateまで。
PR_control_stepは同じKF予測とPRを実行し、Doppler更新を省く未変更control。no_resampleはnativeの
同じ処理でresampleだけを止める。publicは実wrapperの入力処理/同期/転送も含む。

| 粒子数 | 処理 | 旧 (ms) | 新 (ms) | 速度比 |
|---:|:---|---:|---:|---:|
| 2,048 | native_PR_KF_step | 0.4205 | 0.3266 | 1.288× |
| 2,048 | native_PR_KF_no_resample | 0.3560 | 0.2970 | 1.198× |
| 2,048 | native_PR_control_step | 0.2221 | 0.2182 | 1.018× |
| 2,048 | public_PR_KF_step | 0.4565 | 0.3586 | 1.273× |
| 2,048 | public_PR_control_step | 0.2648 | 0.2250 | 1.177× |
| 10,000 | native_PR_KF_step | 0.6381 | 0.6073 | 1.051× |
| 10,000 | native_PR_KF_no_resample | 0.5909 | 0.6092 | 0.970× |
| 10,000 | native_PR_control_step | 0.2650 | 0.2515 | 1.053× |
| 10,000 | public_PR_KF_step | 0.6540 | 0.6356 | 1.029× |
| 10,000 | public_PR_control_step | 0.2949 | 0.2717 | 1.086× |
| 100,000 | native_PR_KF_step | 4.2115 | 4.1534 | 1.014× |
| 100,000 | native_PR_KF_no_resample | 3.9039 | 3.8152 | 1.023× |
| 100,000 | native_PR_control_step | 1.2349 | 1.2304 | 1.004× |
| 100,000 | public_PR_KF_step | 4.2566 | 4.1370 | 1.029× |
| 100,000 | public_PR_control_step | 1.2636 | 1.2421 | 1.017× |
| 1,000,000 | native_PR_KF_step | 41.6686 | 41.5521 | 1.003× |
| 1,000,000 | native_PR_KF_no_resample | 35.0905 | 35.0426 | 1.001× |
| 1,000,000 | native_PR_control_step | 13.4527 | 13.4743 | 0.998× |
| 1,000,000 | public_PR_KF_step | 41.6011 | 41.6155 | 1.000× |
| 1,000,000 | public_PR_control_step | 13.5381 | 13.4991 | 1.003× |

### 未変更control

| 粒子数 | 処理 | 旧 (ms) | 新 (ms) | 速度比 |
|---:|:---|---:|---:|---:|
| 2,048 | Gaussian_control | 0.0646 | 0.0740 | 0.873× |
| 2,048 | resample_control | 0.0305 | 0.0316 | 0.965× |
| 2,048 | predict_control | 0.0337 | 0.0353 | 0.956× |
| 10,000 | Gaussian_control | 0.0945 | 0.0913 | 1.036× |
| 10,000 | resample_control | 0.0398 | 0.0389 | 1.023× |
| 10,000 | predict_control | 0.0582 | 0.0577 | 1.008× |
| 100,000 | Gaussian_control | 0.3846 | 0.3678 | 1.046× |
| 100,000 | resample_control | 0.2326 | 0.2286 | 1.017× |
| 100,000 | predict_control | 0.3951 | 0.3899 | 1.013× |
| 1,000,000 | Gaussian_control | 3.2790 | 3.2576 | 1.007× |
| 1,000,000 | resample_control | 2.0938 | 2.1000 | 0.997× |
| 1,000,000 | predict_control | 3.6043 | 3.6179 | 0.996× |

2,048粒子/12衛星のKF単独round中央値はbaseline0.1964〜0.2439 ms、candidate0.1140〜0.1921 msで範囲が分離した。
公開全体はbaseline0.3968〜0.5485 ms、candidate0.2989〜1.0877 msで重なりがあり、同粒子数の未変更public PR controlも
1.177×と動いた。native全体の未変更PR controlは
1.018×。公開全体の1.273×を固定利得としたり、全てを
KF launchへ帰属させたりしない。10万粒子の単独KF12/32は約1.028×だが、未変更Gaussian controlも
1.046×だった。100万粒子KF32は
70.7367→71.3319 ms（0.992×）と僅かな悪化も記録。大粒子数での大きな利得、実データ位置精度、
whole GNSS・IMU pipelineの利得は未評価。

**2,070ケースで全16状態/log weight/更新軌跡がbit一致**。
1〜100万粒子、0〜128衛星、16/17/32/33のcache境界と63/64/65/127/128/129のblock境界、
通常/非対称/非SPD/zero/非有限/微小・巨大covariance、masked/near/invalid-S/empty/tiny-sigma/
非有限weight、2回更新、観測buffer成長とresample、14件の5epoch PR/KF軌跡を含む。
全72測定条件/全round/各回の返却値hashと各stage最後のstate/log-weight snapshotも一致した。
時間測定中の全回でstate dumpは行っていない。既存systematic Thrust CDF境界変動は範囲外。

関連PF/wrapper/smoother/SMC/IMU **570テストPASS**。追加60件は独立FP64 NumPy行列参照、KF後の
ESS/estimate、full cloudと分割cloudのbit一致、cache/growth/resample混合、返却array所有権を検証。
追加suiteのCUDA **memcheck 0 errors/0 bytes leaked、racecheck 0 hazards/0 errors/0 warnings、
synccheck 0 errors**。

対称積再利用、共分散の読み込み後退、64-thread版も予備測定したが採用しなかった。
各案は1,728ケースでbit一致したものの、大粒子数の安定利得を示せず、前2案のregisterは86→88/90。
こちらは**1round×20回・非交互の予備測定**で採用証拠にしない。raw timings/検証manifest hash/
source・binary hashを [probe JSON](pf_kf_probes_20261012.json)、元の計算ASTと同一の再現コードを
[probe](probes/bench_pf_kf_launch_probe.py)、baselineから各候補sourceを再構築する3 patchをJSONから参照する。
patchは直前予測版に適用して各試行sourceのCRLF bytes/SHAを再現できることを検証した。
選定した128-thread版の採用根拠は上記の5交互round測定とdevice SASS/bit検証。

```powershell
cmake --build build/diag --config Release --target _gnss_gpu_pf_device --parallel 4
$env:PYTHONPATH='python'
python benchmarks/bench_pf_kf_launch_compare.py --baseline dist/pf_kf_symmetry_baseline/_gnss_gpu_pf_device.cp312-win_amd64.pyd --candidate build/diag/Release/_gnss_gpu_pf_device.cp312-win_amd64.pyd --baseline-dir dist/pf_kf_symmetry_baseline --rounds 5 --repeats 20 --output dist/pf_kf_launch_final_20261012.json
python -m pytest tests/test_pf_kf_launch_layout.py -q
```

baseline_dirには直前予測版の5 sourceのbasenameとbinaryをbyte-exact保存したものを指定する。
比較は [script](bench_pf_kf_launch_compare.py)、独立/分割参照は [test](../tests/test_pf_kf_launch_layout.py)。

## PF prediction specialization and captured guides (2026-10-12)

通常・速度ノイズ付き・速度KFの3予測をlaunch時にdispatchし、各modeで不要な処理をcompile時に
除いた。速度ガイド3個のFP64値をkernel引数で捕捉し、毎回の独立した24-byte H2D copyを省いた。
100万粒子の通常native予測は **2.2659→2.1475 ms（1.055×）**、公開wrapperの
8回連続予測は **17.9598→17.0453 ms（1.054×）**（8回全体の時間）。
10万粒子の公開8回連続予測も **2.4208→1.8286 ms（1.324×）**。
公開PR更新全体は **12.1421→12.0994 ms（1.004×）**。
直前Gaussian PR版からの実測差で、過去の速度比を掛け合わせていない。

Philoxのseed/particle/step、normal draw順、FP64、位置/clock更新式、KF Cholesky/clampとQv更新を
維持した。KFはsigma_vel>0でも速度をsampleせず、非KFはsigma_vel>0だけsampleする。
ガイドblendは旧Release binaryの丸め順（alpha*guideをroundしてからprior側をFMA）を
`__dmul_rn`/`__fma_rn`で明示。引数化だけでは逆側の積へFMAが変わり速度に1 ULP差が出たため、
その案は修正してから本測定した。alpha=1でも非有限priorに0を掛ける元の動作を保つ。
ガイドを共有pinned-result bufferへ書かないので、未完の予測がある間の転送元上書きも避けられる。
GPU registerは旧mixed66→通常48/速度ノイズ58/KF60、stack/shared/localは全て0。
追加owner buffer/cudaMalloc/launch/syncは0。既存の24-byte d_vel予約・header/bindings/APIは維持。

baselineの5 source/binary hashを [直前Gaussian PR版](pf_gaussian_gpu_20261012.json) のcandidateに固定。
CUDA12.8/MSVC Release/compute capability7.5/C++20/fast mathなし。
別processの実production extensionを順次実行し、**5round×20回、旧新版順をroundごとに交互**にした。
3 warm-up後の同期込みwall time、表はround中央値の中央値。初期化/入力復元/hash/state dumpは測定外。
host/GPUは非専有。全76条件・raw samples・source/binary/benchmark hashesを
[JSON](pf_predict_gpu_20261012.json) に保存した。

### 単独予測と連続呼び出し

通常はalpha1/sigma_vel0、blendはalpha0.3/sigma_vel0、noisyはalpha0.3/sigma_vel0.2、
KFはalpha0.3/Qv0.1。native_KF_importedは粒子ごとに異なる相関SPD共分散を復元して測定。
queue8は実公開wrapperで8回predictし最後だけ同期する。旧pinned-guide referenceとの比較を
成立させるため、測定では全8回に同一guideを渡す。異なるguideの非同期列は新unit testで検証した。

| 粒子数 | 処理 | 旧 (ms) | 新 (ms) | 速度比 |
|---:|:---|---:|---:|---:|
| 2,048 | native_predict_default | 0.0483 | 0.0389 | 1.242× |
| 2,048 | native_predict_blend | 0.0485 | 0.0356 | 1.362× |
| 2,048 | native_predict_noisy | 0.0617 | 0.0500 | 1.234× |
| 2,048 | native_predict_KF | 0.0733 | 0.0465 | 1.577× |
| 2,048 | public_queue8_predict | 0.6435 | 0.3726 | 1.727× |
| 10,000 | native_predict_default | 0.0874 | 0.0529 | 1.651× |
| 10,000 | native_predict_blend | 0.0855 | 0.0553 | 1.548× |
| 10,000 | native_predict_noisy | 0.1313 | 0.0809 | 1.624× |
| 10,000 | native_predict_KF | 0.1075 | 0.0724 | 1.485× |
| 10,000 | public_queue8_predict | 0.7081 | 0.3545 | 1.997× |
| 100,000 | native_predict_default | 0.3122 | 0.2982 | 1.047× |
| 100,000 | native_predict_blend | 0.3158 | 0.2992 | 1.056× |
| 100,000 | native_predict_noisy | 0.5658 | 0.5494 | 1.030× |
| 100,000 | native_predict_KF | 0.5412 | 0.4858 | 1.114× |
| 100,000 | public_queue8_predict | 2.4208 | 1.8286 | 1.324× |
| 1,000,000 | native_predict_default | 2.2659 | 2.1475 | 1.055× |
| 1,000,000 | native_predict_blend | 2.2515 | 2.1484 | 1.048× |
| 1,000,000 | native_predict_noisy | 4.1167 | 4.0694 | 1.012× |
| 1,000,000 | native_predict_KF | 3.7074 | 3.5945 | 1.031× |
| 1,000,000 | public_queue8_predict | 17.9598 | 17.0453 | 1.054× |

### 12衛星PR更新全体

predict→PR→任意のDoppler KF→ESS→条件付きB60 resample→estimateを測定。公開wrapperは実入力処理と
転送も含む。PR_KFはDoppler尤度も合成してからresample。high_ESSはPR sigma1000でresampleを回避。

| 粒子数 | 処理 | 旧 (ms) | 新 (ms) | 速度比 |
|---:|:---|---:|---:|---:|
| 2,048 | native_PR_step | 0.2573 | 0.2508 | 1.026× |
| 2,048 | native_PR_noisy_step | 0.2565 | 0.2680 | 0.957× |
| 2,048 | native_PR_KF_step | 0.5799 | 0.4781 | 1.213× |
| 2,048 | native_PR_high_ESS_step | 0.2297 | 0.2714 | 0.846× |
| 2,048 | public_PR_default | 0.5196 | 0.3027 | 1.717× |
| 2,048 | public_PR_noisy | 0.3242 | 0.3684 | 0.880× |
| 2,048 | public_PR_KF | 0.6977 | 0.6830 | 1.022× |
| 10,000 | native_PR_step | 0.3456 | 0.3206 | 1.078× |
| 10,000 | native_PR_noisy_step | 0.3871 | 0.3377 | 1.147× |
| 10,000 | native_PR_KF_step | 0.7743 | 0.7824 | 0.990× |
| 10,000 | native_PR_high_ESS_step | 0.2656 | 0.2485 | 1.068× |
| 10,000 | public_PR_default | 0.5050 | 0.3625 | 1.393× |
| 10,000 | public_PR_noisy | 0.3663 | 0.3906 | 0.938× |
| 10,000 | public_PR_KF | 0.8607 | 0.8061 | 1.068× |
| 100,000 | native_PR_step | 1.3058 | 1.3454 | 0.971× |
| 100,000 | native_PR_noisy_step | 1.4431 | 1.6434 | 0.878× |
| 100,000 | native_PR_KF_step | 4.7895 | 4.9803 | 0.962× |
| 100,000 | native_PR_high_ESS_step | 0.8874 | 0.8768 | 1.012× |
| 100,000 | public_PR_default | 1.1346 | 1.2430 | 0.913× |
| 100,000 | public_PR_noisy | 1.3250 | 1.3939 | 0.951× |
| 100,000 | public_PR_KF | 4.4845 | 4.2788 | 1.048× |
| 1,000,000 | native_PR_step | 12.1445 | 12.0513 | 1.008× |
| 1,000,000 | native_PR_noisy_step | 14.0176 | 14.2465 | 0.984× |
| 1,000,000 | native_PR_KF_step | 41.4905 | 41.5712 | 0.998× |
| 1,000,000 | native_PR_high_ESS_step | 6.4617 | 6.4352 | 1.004× |
| 1,000,000 | public_PR_default | 12.1421 | 12.0994 | 1.004× |
| 1,000,000 | public_PR_noisy | 14.0572 | 14.0107 | 1.003× |
| 1,000,000 | public_PR_KF | 41.5510 | 41.8008 | 0.994× |

### 未変更control

| 粒子数 | 処理 | 旧 (ms) | 新 (ms) | 速度比 |
|---:|:---|---:|---:|---:|
| 2,048 | Gaussian_control | 0.1085 | 0.1162 | 0.934× |
| 2,048 | resample_control | 0.0500 | 0.0381 | 1.314× |
| 2,048 | KF_control | 0.2210 | 0.2405 | 0.919× |
| 10,000 | Gaussian_control | 0.0980 | 0.1018 | 0.964× |
| 10,000 | resample_control | 0.0479 | 0.0486 | 0.984× |
| 10,000 | KF_control | 0.4553 | 0.4517 | 1.008× |
| 100,000 | Gaussian_control | 0.3893 | 0.4111 | 0.947× |
| 100,000 | resample_control | 0.2311 | 0.2331 | 0.991× |
| 100,000 | KF_control | 2.8969 | 2.9212 | 0.992× |
| 1,000,000 | Gaussian_control | 3.2325 | 3.2467 | 0.996× |
| 1,000,000 | resample_control | 2.1006 | 2.1211 | 0.990× |
| 1,000,000 | KF_control | 26.9010 | 26.9159 | 0.999× |

100万粒子native通常予測のround中央値はbaseline2.2107〜2.2920 ms、candidate2.1295〜2.1894 ms。
公開PR全体はbaseline12.0652〜12.3183 ms、candidate12.0358〜12.3040 ms。
小粒子数のhost変動と未変更controlの変動も残しており、単一roundの大きな速度比を採用根拠にしない。
10万粒子public PRは1.1346→1.2430 ms（0.913×）と悪化し、同粒子数Gaussian controlも
0.947×だった。100万粒子public PR/KFもほぼ差がなく、
全更新の安定した利得はこの測定では示せない。一方、10万粒子queue8は
baseline2.3224〜2.7138 ms、candidate1.8085〜1.8880 msで、100万粒子通常予測/queue8と共に旧新版のround範囲が分離した。
大粒子数の速度ノイズ/KFや更新全体の利得は表の通り限定的。実データ位置精度、whole GNSS・IMU
pipelineの利得は未評価。systematic resamplerの既存Thrust CDF境界変動は範囲外。

**1,836ケースの全16状態/log weight/更新軌跡が旧新版でbit一致**。
1/31/32/33/255/256/257/8193粒子、alpha範囲外/境界/nextafter、負sigma_vel、zero dt/負Qv、
微小/大sigma、small/ECEF座標、相関SPD/zero/非SPD/非対称/非有限共分散、非有限速度を含む。
10万/100万粒子の通常/速度ノイズ/KF予測と、16件の5epoch PR/PR_noisy/PR_KF/high_ESS軌跡も検証。
全測定条件/全round/各回の返却値hashと各stage最後のstate/log-weight snapshotも一致した。
時間測定中の全回でstate dumpは行っていない。

関連PF/wrapper/smoother/SMC/IMU **510テストPASS**。追加78件は独立分布moment/covariance参照、
zero-dt blend、固定seed replay、log weight/返却array所有権、非有限prior、負Qvと、
通常/noisy/KFを切り替えguideを毎回変える48回非同期列（native/public）を検証。
追加suiteはCUDA **memcheck 0 errors/0 bytes leaked、racecheck 0 hazards/0 errors/0 warnings**。

```powershell
cmake --build build/diag --config Release --target _gnss_gpu_pf_device --parallel 4
$env:PYTHONPATH='python'
python benchmarks/bench_pf_predict_compare.py --baseline dist/pf_predict_baseline/_gnss_gpu_pf_device.cp312-win_amd64.pyd --candidate build/diag/Release/_gnss_gpu_pf_device.cp312-win_amd64.pyd --baseline-dir dist/pf_predict_baseline --rounds 5 --repeats 20 --output dist/pf_predict_final_20261012.json
python -m pytest tests/test_pf_predict_specialization.py -q
```

baseline_dirは直前版の5 sourceのbasenameとbinaryをbyte-exact保存したものを指定する。
再現に必要なbuild/source anchorと実測raw sampleは公開JSON、probe/validationは
[比較script](bench_pf_predict_compare.py)、追加検証は [test](../tests/test_pf_predict_specialization.py)。

## Default Gaussian PR specialization (2026-10-12)

既定のGaussian pseudorange尤度を専用CUDA kernelへ分け、GPU上で粒子共通の逆分散をblock内で
一度計算し、衛星loopを2件ずつunrollした。100万粒子/12衛星の単独weightは
**3.5309→3.2065 ms（1.10×）**。公開predict/PR/ESS/条件付きB60 resample/estimate全体は
**12.3487→12.0579 ms（1.024×、約2.4%短縮）**。resampleしない高ESSのnative stepは
**6.6986→6.3755 ms（1.05×、約4.8%短縮）**。4衛星は単独1.18×、64/65衛星は1.06×。
直前のquery-reduction版との実測差で、過去の速度比を掛け合わせていない。

専用経路の条件はnu<=0、particle gate無効、Huber無効（またはk<=0）。Student/Huber/正のgateは
元のgeneral kernelで実行し、Student優先・gateのmin_kept fallbackなどの既存動作を維持する。
rangeのsqrt、residual、`-0.5 * ws * residual * residual * inv_sigma2`、衛星の累積順、FP64精度、
log weightのoverwriteを同じにした。逆分散をCPUへ移さず、元のGPU式`1.0/(sigma*sigma)`で
計算する。2件のunrollは同じ各衛星の演算・加算順を保ち、奇数衛星の末尾も処理する。
入力検証・3 H2D copy・stream同期・launch数/block256・workspace再利用・公開APIも同じ。

追加のGPU owner buffer/cudaMallocは0。新kernelのcompiled registerは38（旧general44）、
stack/localは0。逆分散1値のためstatic sharedがcompiledで16 bytes増え、dynamicは従来の
n_sat*5*8 bytes。1,228衛星では49,120+16=49,136 bytesとなり48 KiBに収まり、既存の最大
shared条件も検証した。general kernelは44 registers/stack0/local0のまま。

baselineの5 source/binary hashを [直前query版](pf_reduction_gpu_20261012.json) のcandidateへ固定。
CUDA 12.8 / MSVC Release / compute capability 7.5 / C++20 / fast mathなし。
別processの実production extensionを順次実行し、**5round×20回、roundごとに旧新版順を交互**にした。
3 warm-up後の同期込みwall time、表はround中央値の中央値。初期化・入力復元・出力hash/state
dumpは測定外。GPU/hostはdesktopと共用で非専有。単独PRは衛星方向・非unit weightの固定入力。
更新stepは12衛星PR→任意のDoppler KFの両尤度→ESS→条件付きB60 resample→estimateまでで、
publicは実Python wrapperの入力処理とhost/GPU転送も含む。high_ESSはsigma_pr=1000。
Student(nu3)/Huber(k1.5)/gate(20m)のcontrolは専用経路を通らない。resample controlはuniform weight。
全68条件とraw sampleを [JSON](pf_gaussian_gpu_20261012.json) に保存した。

### Gaussian PR

| 粒子数 | 処理 | 旧 (ms) | 新 (ms) | 速度比 |
|---:|:---|---:|---:|---:|
| 2,048 | Gaussian_sat4 | 0.0461 | 0.0457 | 1.01× |
| 2,048 | Gaussian_sat12 | 0.0604 | 0.0625 | 0.96× |
| 2,048 | Gaussian_sat16 | 0.0779 | 0.0751 | 1.04× |
| 2,048 | Gaussian_sat32 | 0.1041 | 0.1014 | 1.03× |
| 2,048 | Gaussian_sat64 | 0.1575 | 0.1526 | 1.03× |
| 2,048 | Gaussian_sat65 | 0.1586 | 0.1545 | 1.03× |
| 10,000 | Gaussian_sat4 | 0.0680 | 0.0556 | 1.22× |
| 10,000 | Gaussian_sat12 | 0.0917 | 0.0809 | 1.13× |
| 10,000 | Gaussian_sat16 | 0.0983 | 0.0950 | 1.03× |
| 10,000 | Gaussian_sat32 | 0.1516 | 0.1442 | 1.05× |
| 10,000 | Gaussian_sat64 | 0.2556 | 0.2434 | 1.05× |
| 10,000 | Gaussian_sat65 | 0.2584 | 0.2484 | 1.04× |
| 100,000 | Gaussian_sat4 | 0.1850 | 0.1491 | 1.24× |
| 100,000 | Gaussian_sat12 | 0.3997 | 0.3665 | 1.09× |
| 100,000 | Gaussian_sat16 | 0.5099 | 0.4738 | 1.08× |
| 100,000 | Gaussian_sat32 | 0.9636 | 0.9011 | 1.07× |
| 100,000 | Gaussian_sat64 | 1.9323 | 1.8193 | 1.06× |
| 100,000 | Gaussian_sat65 | 1.9578 | 1.8489 | 1.06× |
| 1,000,000 | Gaussian_sat4 | 1.3531 | 1.1495 | 1.18× |
| 1,000,000 | Gaussian_sat12 | 3.5309 | 3.2065 | 1.10× |
| 1,000,000 | Gaussian_sat16 | 4.6186 | 4.2415 | 1.09× |
| 1,000,000 | Gaussian_sat32 | 8.9632 | 8.3498 | 1.07× |
| 1,000,000 | Gaussian_sat64 | 17.6356 | 16.5879 | 1.06× |
| 1,000,000 | Gaussian_sat65 | 17.9111 | 16.8787 | 1.06× |

### 12衛星の更新全体

| 粒子数 | 処理 | 旧 (ms) | 新 (ms) | 速度比 |
|---:|:---|---:|---:|---:|
| 2,048 | native_PR_step | 0.1931 | 0.1925 | 1.00× |
| 2,048 | native_PR_KF_step | 0.3896 | 0.3905 | 1.00× |
| 2,048 | native_PR_high_ESS_step | 0.1764 | 0.1622 | 1.09× |
| 2,048 | public_PR_step | 0.2311 | 0.2328 | 0.99× |
| 2,048 | public_PR_KF_step | 0.4148 | 0.4447 | 0.93× |
| 10,000 | native_PR_step | 0.2412 | 0.2386 | 1.01× |
| 10,000 | native_PR_KF_step | 0.6101 | 0.6196 | 0.98× |
| 10,000 | native_PR_high_ESS_step | 0.2318 | 0.2206 | 1.05× |
| 10,000 | public_PR_step | 0.3822 | 0.2494 | 1.53× |
| 10,000 | public_PR_KF_step | 0.6429 | 0.6289 | 1.02× |
| 100,000 | native_PR_step | 1.1004 | 1.0645 | 1.03× |
| 100,000 | native_PR_KF_step | 4.1851 | 4.1774 | 1.00× |
| 100,000 | native_PR_high_ESS_step | 0.8195 | 0.7885 | 1.04× |
| 100,000 | public_PR_step | 1.1004 | 1.0791 | 1.02× |
| 100,000 | public_PR_KF_step | 4.2024 | 4.1683 | 1.01× |
| 1,000,000 | native_PR_step | 12.3444 | 12.0123 | 1.03× |
| 1,000,000 | native_PR_KF_step | 41.5002 | 41.1682 | 1.01× |
| 1,000,000 | native_PR_high_ESS_step | 6.6986 | 6.3755 | 1.05× |
| 1,000,000 | public_PR_step | 12.3487 | 12.0579 | 1.02× |
| 1,000,000 | public_PR_KF_step | 41.4933 | 41.1946 | 1.01× |

### 既存robust経路と未変更control

| 粒子数 | 処理 | 旧 (ms) | 新 (ms) | 速度比 |
|---:|:---|---:|---:|---:|
| 2,048 | Student_control | 0.1008 | 0.1035 | 0.97× |
| 2,048 | Huber_control | 0.0844 | 0.0846 | 1.00× |
| 2,048 | Gate_control | 0.0914 | 0.0874 | 1.05× |
| 2,048 | predict_control | 0.0360 | 0.0360 | 1.00× |
| 2,048 | resample_control | 0.0301 | 0.0301 | 1.00× |
| 2,048 | KF_control | 0.1948 | 0.1959 | 0.99× |
| 10,000 | Student_control | 0.1437 | 0.1451 | 0.99× |
| 10,000 | Huber_control | 0.1129 | 0.1111 | 1.02× |
| 10,000 | Gate_control | 0.1162 | 0.1171 | 0.99× |
| 10,000 | predict_control | 0.0481 | 0.0485 | 0.99× |
| 10,000 | resample_control | 0.0374 | 0.0374 | 1.00× |
| 10,000 | KF_control | 0.3552 | 0.3555 | 1.00× |
| 100,000 | Student_control | 0.9012 | 0.8984 | 1.00× |
| 100,000 | Huber_control | 0.6252 | 0.6263 | 1.00× |
| 100,000 | Gate_control | 0.6719 | 0.6804 | 0.99× |
| 100,000 | predict_control | 0.2444 | 0.2446 | 1.00× |
| 100,000 | resample_control | 0.2235 | 0.2239 | 1.00× |
| 100,000 | KF_control | 2.8735 | 2.8787 | 1.00× |
| 1,000,000 | Student_control | 8.3521 | 8.3556 | 1.00× |
| 1,000,000 | Huber_control | 5.7206 | 5.7294 | 1.00× |
| 1,000,000 | Gate_control | 6.1677 | 6.1754 | 1.00× |
| 1,000,000 | predict_control | 2.2262 | 2.2252 | 1.00× |
| 1,000,000 | resample_control | 2.0844 | 2.0854 | 1.00× |
| 1,000,000 | KF_control | 26.6966 | 26.6947 | 1.00× |

100万粒子12衛星の単独PRのround中央値は旧3.5160〜3.5639 ms、新3.2014〜3.2226 ms。
公開PR全体も旧12.3337〜12.4625 ms、新12.0318〜12.1793 msだった。100万粒子の未変更
predict/resample/KFとStudent/Huber/gateは0.998〜1.001×程度で、PR+KF全体は1.007×と小さい。
小粒子数ではhost変動があり、2,048粒子Gaussian12は0.96×、公開PR+KFは0.93×などの悪化も
記録する。10,000粒子の公開PRの1.53×はnative全体1.01×との乖離があり、代表的な利得として
扱わない。すべての条件の高速化、実データ位置精度、whole GNSS・IMU pipelineの利得は未評価。

**1,205ケースの全16状態・log weight・返却値/更新軌跡hashが旧新版でbit一致**。
1〜100万粒子、1〜128衛星の奇数/偶数・shared/growth境界、通常/負nu/無効Huber/zero weight/
sigma² underflow・overflow/非有限粒子/近接・zero rangeと、Student/Huber/gateの既存経路を含む。
各weight call後の状態不変、weightのoverwrite/replay、模式切替とbuffer growth、PR/PR+KF/
高ESSの12件の5epoch軌跡を検証した。全68測定条件・全round・各回の返却値hashと、各stage
最後のstate/log-weight snapshotも一致（時間測定中の全回でstate dumpは行っていない）。
resampler/乱数/Thrust CDFは従来のままで、systematicの既存境界変動の解消は範囲外。

関連PF/wrapper/smoother/SMC/IMUの**432テストPASS**。追加89件は独立NumPy参照
（rtol5e-9/atol1e-8）、同じGaussian尤度を元のgeneral kernelで計算したbit照合、奇数衛星、
端数粒子、上書き/返却配列所有権/状態不変、極小・極大sigma、無効robust optionと48 KiB境界/
成長後の再利用を検証する。CUDA memcheckは89件で0 errors/0 bytes leaked、racecheckも
0 hazards/0 errors/0 warnings。

同じ構成の直前baselineを退避したローカル例:

```powershell
$env:PYTHONPATH = "python"
python benchmarks/bench_pf_gaussian_compare.py `
  --baseline dist/pf_gaussian_baseline/_gnss_gpu_pf_device.cp312-win_amd64.pyd `
  --candidate build/diag/Release/_gnss_gpu_pf_device.cp312-win_amd64.pyd `
  --baseline-dir dist/pf_gaussian_baseline `
  --rounds 5 --repeats 20 --output dist/pf_gaussian_recheck.json
```

source/extension/benchmark hashはJSONに記録。
[比較スクリプト](bench_pf_gaussian_compare.py)、[検証テスト](../tests/test_pf_gaussian_weight.py) を参照。

### 採用しなかったresampling案

先にproposal列を別kernelで生成/既存scratchへ共有する案と、weight先読み4→8件の案を試した。
8件案は予備測定で改善して見えたが、production binaryの5交互round×20回では安定した利得が
なく採用しなかった。100,000粒子normal/common-cov B60は0.2659→0.2696 ms（0.99×）、B240は
0.7510→0.7557 ms（0.99×）、100万粒子の公開PRも12.4286→12.3849 ms（1.004×）だった。
4,558ケースの入力/状態/軌跡hashはbit一致したが、register42→51となるtradeoffもあり、
productionのresamplerは直前の4件先読み版を維持する。
[棄却案の全72条件/raw sample](pf_prefetch8_rejected_20261012.json)、
[candidate patch](probes/pf_prefetch8.patch)、[比較スクリプト](bench_pf_prefetch_compare.py) を保存した。
patchはquery-reduction版のsourceに対するもので、reportのcandidate hashはその棄却snapshot。
現在のproduction source/binaryを表すのはGaussian JSONのcandidateである。

## Ordered FP64 PF query reductions (2026-10-12)

ESS・weighted estimate・位置のRMS spreadの集計で、256thread reduction treeの最後の5段を
warp shuffleとレジスタへ移した。100万粒子・normal weightの単独queryはESS
**0.4884→0.4074 ms（1.20×）**、estimate **0.5948→0.5361 ms（1.11×）**、spread
**0.5746→0.5089 ms（1.13×）**。公開wrapperのESS/estimate/default-center spreadの一式は
**2.4147→2.0137 ms（1.20×、約16.6%短縮）**。直前のDoppler-KF版との実測差で、過去の
速度比を掛け合わせていない。

shared-memoryのstride 128/64/32と、その後のwarp stride 16/8/4/2/1の各加算nodeを維持した。
warpの32laneすべてでshuffleし、その段で必要なlaneだけ値を更新する。block maxの後は
全threadで同期してからexpへ進む。最大値・sumの各reductionはblock同期8→4回となり、
6種類のquery/finalize kernelそれぞれで計8回減る。粒子ごとのexp/乗算、partialの読み順、
FP64精度、最後のdivision/spread guard、host copy/stream同期、block数と256threadは同じ。
resamplingのtree/CDF・乱数・観測処理には変更を加えていない。

追加のGPU owner buffer/cudaMallocはない。dynamic sharedはESS/spreadの6,144 bytesと
estimateの12,288 bytesを維持する。cuobjdumpのquery kernel register数はESS20→19、
estimate28→30、spread20→22。finalizeは24/38/24のままで、6 kernelすべてcompiled
stack/localは0。workspace保持量を増やさず、block同期と最後の5段のshared load/storeを減らす。

baselineの5 sourceとbinary hashを [直前Doppler-KF版](pf_doppler_gpu_20261012.json) の
candidateと照合。CUDA 12.8 / MSVC Release / compute capability 7.5 / C++20 / fast mathなし。
別processの実production extensionを順次実行し、**5round×20回、roundごとに旧新版順を交互**
にした。3 warm-up後の同期込みwall time、表はround中央値の中央値。復元・初期化・出力hash
検査は測定外。GPU/hostはdesktopと共用で非専有。

native_diagnosticsはnormal weightでESS/estimate/固定center spreadの3照会。
public_diagnosticsは公開get_ess/estimate/get_position_spread(center=None)を実行し、最後の
spreadが内部でestimateを再取得するため4照会を含む。公開診断の初期重みはuniform。
normalはseed固定のlog weight、uniformは
すべて0。更新stepはpredict→PR（12衛星）→任意のDoppler KF→ESS→条件付きB60 coalesced
resample→estimateで、spread付きは返却mean中心のRMSも含む。no_resampleはsigma_pr=1000の
高ESS条件。publicは実Python wrapperで、入力処理とhost/GPU転送も含む。
全60条件とraw sampleを [JSON](pf_reduction_gpu_20261012.json) に保存した。

### 単独queryと診断集計

| 粒子数 | 処理 | 旧 (ms) | 新 (ms) | 速度比 |
|---:|:---|---:|---:|---:|
| 2,048 | ESS_normal | 0.0471 | 0.0374 | 1.26× |
| 2,048 | estimate_normal | 0.0582 | 0.0490 | 1.19× |
| 2,048 | spread_normal | 0.0551 | 0.0558 | 0.99× |
| 2,048 | ESS_uniform | 0.0475 | 0.0419 | 1.13× |
| 2,048 | estimate_uniform | 0.0559 | 0.0451 | 1.24× |
| 2,048 | spread_uniform | 0.0619 | 0.0454 | 1.36× |
| 2,048 | native_diagnostics | 0.1631 | 0.1363 | 1.20× |
| 2,048 | public_diagnostics | 0.2096 | 0.1957 | 1.07× |
| 10,000 | ESS_normal | 0.0512 | 0.0371 | 1.38× |
| 10,000 | estimate_normal | 0.0579 | 0.0640 | 0.90× |
| 10,000 | spread_normal | 0.0600 | 0.0446 | 1.35× |
| 10,000 | ESS_uniform | 0.0527 | 0.0462 | 1.14× |
| 10,000 | estimate_uniform | 0.0606 | 0.0537 | 1.13× |
| 10,000 | spread_uniform | 0.0653 | 0.0447 | 1.46× |
| 10,000 | native_diagnostics | 0.1694 | 0.1527 | 1.11× |
| 10,000 | public_diagnostics | 0.2355 | 0.2223 | 1.06× |
| 100,000 | ESS_normal | 0.0984 | 0.0908 | 1.08× |
| 100,000 | estimate_normal | 0.1336 | 0.1216 | 1.10× |
| 100,000 | spread_normal | 0.1108 | 0.1128 | 0.98× |
| 100,000 | ESS_uniform | 0.0998 | 0.1010 | 0.99× |
| 100,000 | estimate_uniform | 0.1278 | 0.1169 | 1.09× |
| 100,000 | spread_uniform | 0.1250 | 0.1271 | 0.98× |
| 100,000 | native_diagnostics | 0.4101 | 0.3096 | 1.32× |
| 100,000 | public_diagnostics | 0.5043 | 0.3947 | 1.28× |
| 1,000,000 | ESS_normal | 0.4884 | 0.4074 | 1.20× |
| 1,000,000 | estimate_normal | 0.5948 | 0.5361 | 1.11× |
| 1,000,000 | spread_normal | 0.5746 | 0.5089 | 1.13× |
| 1,000,000 | ESS_uniform | 0.5826 | 0.4195 | 1.39× |
| 1,000,000 | estimate_uniform | 0.7459 | 0.6496 | 1.15× |
| 1,000,000 | spread_uniform | 0.7127 | 0.6183 | 1.15× |
| 1,000,000 | native_diagnostics | 2.0269 | 1.7595 | 1.15× |
| 1,000,000 | public_diagnostics | 2.4147 | 2.0137 | 1.20× |

### 12衛星の更新step

| 粒子数 | 処理 | 旧 (ms) | 新 (ms) | 速度比 |
|---:|:---|---:|---:|---:|
| 2,048 | native_PR_step | 0.2525 | 0.2296 | 1.10× |
| 2,048 | native_PR_no_resample_step | 0.2229 | 0.1978 | 1.13× |
| 2,048 | native_PR_KF_step | 0.4846 | 0.4024 | 1.20× |
| 2,048 | native_PR_spread_step | 0.3216 | 0.2565 | 1.25× |
| 2,048 | public_PR_step | 0.3391 | 0.2623 | 1.29× |
| 2,048 | public_PR_KF_step | 0.5155 | 0.4412 | 1.17× |
| 10,000 | native_PR_step | 0.3300 | 0.2861 | 1.15× |
| 10,000 | native_PR_no_resample_step | 0.2776 | 0.2359 | 1.18× |
| 10,000 | native_PR_KF_step | 0.7836 | 0.6848 | 1.14× |
| 10,000 | native_PR_spread_step | 0.3921 | 0.3448 | 1.14× |
| 10,000 | public_PR_step | 0.3176 | 0.3384 | 0.94× |
| 10,000 | public_PR_KF_step | 0.8259 | 0.7511 | 1.10× |
| 100,000 | native_PR_step | 1.4451 | 1.3407 | 1.08× |
| 100,000 | native_PR_no_resample_step | 1.1285 | 1.0322 | 1.09× |
| 100,000 | native_PR_KF_step | 4.3377 | 4.8955 | 0.89× |
| 100,000 | native_PR_spread_step | 1.2813 | 1.2182 | 1.05× |
| 100,000 | public_PR_step | 1.2704 | 1.1224 | 1.13× |
| 100,000 | public_PR_KF_step | 4.4211 | 4.2389 | 1.04× |
| 1,000,000 | native_PR_step | 12.6392 | 12.4049 | 1.02× |
| 1,000,000 | native_PR_no_resample_step | 6.9509 | 6.7410 | 1.03× |
| 1,000,000 | native_PR_KF_step | 41.9556 | 41.7887 | 1.00× |
| 1,000,000 | native_PR_spread_step | 13.3360 | 12.8943 | 1.03× |
| 1,000,000 | public_PR_step | 12.6661 | 12.3835 | 1.02× |
| 1,000,000 | public_PR_KF_step | 41.9750 | 41.8294 | 1.00× |

### 未変更PR weight control

| 粒子数 | 処理 | 旧 (ms) | 新 (ms) | 速度比 |
|---:|:---|---:|---:|---:|
| 2,048 | weight_control | 0.0766 | 0.0738 | 1.04× |
| 10,000 | weight_control | 0.1047 | 0.0996 | 1.05× |
| 100,000 | weight_control | 0.5375 | 0.4957 | 1.08× |
| 1,000,000 | weight_control | 4.4793 | 4.0641 | 1.10× |

100万粒子の公開PR更新全体は12.6661→12.3835 ms（1.02×、約2.2%差）、PR+KFは
41.9750→41.8294 ms（1.003×）で差が小さい。未変更weight controlも4.4793→4.0641 ms
（1.10×）と変動し、旧round中央値は3.9174〜5.8035 ms、新は3.9813〜4.4698 msだった。
10,000粒子の公開PRは0.94×、100,000粒子のnative PR+KFは0.89×など、遅く測定された条件も
含めて公開する。100万粒子の公開診断集計のround中央値は旧2.3193〜2.4710 ms、新
1.9706〜2.0721 ms。集計の短縮は繰り返し確認したが、更新全体の小さい差を安定した利得と
断定しない。実データ位置精度/whole GNSS・IMU pipelineの利得は未評価。

**2,372検証ケース**（2,200 query、160 unchanged-state、12件の5epoch軌跡）のうち
**2,281件は出力hashがbit一致**。残り91件も、異なる値は両版ともNaNで差は符号bitだけ。
NaN以外の値はすべてbit一致し、全state/log-weight snapshotと全軌跡も一致した。
NaNを含む入力の符号/payloadの一律bit再現性は約束しない。各caseの入力・出力hashとqueryの
生bit列、NaN差のcase名をJSONに残し、差異を保持して比較した。
1〜100万粒子のwarp/block/finalize端数、14種類の重み（大shift/極端差/±Inf/NaN/単一有限値）、
ECEF/符号付き0/相殺/微小/非有限/巨大座標を含む。
**全60測定条件・全round・各回の返却値hashも一致**し、各stage最後の全16状態/log weightの
snapshotも一致した（時間測定中の全回でstate dumpを行ったわけではない）。

関連PF/wrapper/smoother/SMC/IMUの**343テストPASS**。追加129テストは独立NumPyの
weighted mean/ESS/RMS参照（rtol5e-13、atolはmean1e-9/ESS1e-10/RMS1e-12）、query順変更後の
返却配列所有権・状態不変・NaN/spread guard・公開resample threshold/default centerを検証する。
129件のCUDA memcheckは0 errors/0 bytes leaked、racecheckは0 hazards/0 errors/0 warnings、
synccheckも0 errors。今回の範囲外にあるsystematic Thrust CDFの既存境界変動は維持される。

同じ構成の直前baselineを退避したローカル例:

```powershell
$env:PYTHONPATH = "python"
python benchmarks/bench_pf_reduction_compare.py `
  --baseline dist/pf_reduction_baseline/_gnss_gpu_pf_device.cp312-win_amd64.pyd `
  --candidate build/diag/Release/_gnss_gpu_pf_device.cp312-win_amd64.pyd `
  --baseline-dir dist/pf_reduction_baseline `
  --rounds 5 --repeats 20 --output dist/pf_reduction_recheck.json
```

source/extension/benchmark hashはJSONに記録。
[比較スクリプト](bench_pf_reduction_compare.py)、
[検証テスト](../tests/test_pf_query_reduction.py) を参照。

## Reusable signal work buffers (2026-10-12)

IQ生成の数式を変更せず、シミュレータごとにGPU出力・航法bit・32channel超のdescriptor領域と
CPU packing vectorのcapacityを再利用した。26万sample・12可視衛星の都市simulation全体は、
1,200 triangleで **2.5379→1.8947 ms（1.34×、約25.3%短縮）**、12,000 triangleで
**3.3476→2.7244 ms（1.23×、約18.6%短縮）**。12 triangleも2.4920→1.8720 ms（1.33×）。
公開SignalSimulatorの12channel/26万sampleは7.0985→5.9108 ms（1.20×）。
これは直前のfused IQ版からの差で、過去の速度比を掛け合わせていない。

`DeviceSignalWorkspace` は毎回の入力をpack/uploadし直し、配列の内容や設定変更を反映する。
3種類のCUDA kernelのbody、float丸め/加算順、navigation bit clamp、固定seedのPhilox系列、
seed=0のfresh noise選択、constellation/PRNの扱いは旧版のまま。device-wide同期とhost IQ copyも
維持した。再利用時はcapacityが足りればcudaMalloc/freeを行わない。既存のnative関数は
一時ownerで単発生成でき、Pythonのgenerate_epoch/generate_test_signalの引数は同じ。
公開wrapperは遅延生成したownerを利用し、古いfunction-only extensionにもfallbackする。
`release_gpu()` で全work bufferを解放でき、その後の生成で再作成する。copy/pickleはGPU ownerを
共有せず、戻り値のNumPy配列も独立storageを持つ。close/destructorはowning deviceで解放する。

保持メモリとのtradeoffがある。outputは最大requested sample数×8 bytesで、26万sampleなら
**2,080,000 bytes**、260万sampleなら**20,800,000 bytes**をrelease/GCまで保持する。
航法bitとdescriptorも最大capacityを保持する（測定の65channel/nav/26万sampleでは合計
2,086,060 bytes）。JSONのretained_device_bytesはownerの3 bufferだけのcapacityを数え、
既存global code table、CPU vector、driver領域、grow時の新旧buffer同時生存などのpeak memoryを
含めない。global code tableの既存単一device前提を維持し、同じinstanceの並行呼出しは対象外。

baselineのsignal source/binary hashを [直前fused版](signal_fused_gpu_20261011.json) のcandidateと照合。
CUDA 12.8 / MSVC Release / compute capability 7.5 / C++20 / fast mathなし。
旧新版の実際のproduction extensionを同一processで読み、**5round×20回、各callの順序を交互**に
した。3 warm-up後の同期込みwall time、表はround中央値の中央値。GPU/hostはdesktopと共用で
非専有。長さやchannel数の異なる条件間で絶対時間を単純比較せず、各条件の旧新版を比較する。
public/urban比較用のnative module切替は双方同じ方法で行い、その小さいhost overheadも含む。
code table初期化、CPU BVH build、出力hash検査は測定外。

native warmは入力検証/変換/packing/upload/kernel/device同期/host IQ取得まで含む。
cold controlは従来の単発APIで、owner生成・確保・解放も含む。
public warmは実際のPython wrapperの入力検証も含み、urban warmは同じresident BVHでの
LOS/反射照会、channel構築と実際のCUDA IQ生成・metadata返却まで含む。BVH binary/geometry側は
今回変更していない。全54条件と生sampleを [JSON](signal_workspace_gpu_20261012.json) に保存した。

### Native warm

| sample数 | channel数 | 航法bit配列 | 旧 (ms) | 新 (ms) | 速度比 |
|---:|---:|:---:|---:|---:|---:|
| 2,600 | 0 | なし | 0.0519 | 0.0445 | 1.17× |
| 2,600 | 1 | なし | 0.0600 | 0.0519 | 1.15× |
| 2,600 | 11 | なし | 0.0813 | 0.0749 | 1.08× |
| 2,600 | 32 | なし | 0.1449 | 0.1380 | 1.05× |
| 26,000 | 0 | なし | 0.0772 | 0.0706 | 1.09× |
| 26,000 | 1 | なし | 0.0880 | 0.0806 | 1.09× |
| 26,000 | 11 | なし | 0.1420 | 0.1358 | 1.05× |
| 26,000 | 32 | なし | 0.2549 | 0.2475 | 1.03× |
| 260,000 | 0 | なし | 1.4037 | 0.8217 | 1.71× |
| 260,000 | 1 | なし | 1.6238 | 0.9805 | 1.66× |
| 260,000 | 11 | なし | 1.8531 | 1.2399 | 1.49× |
| 260,000 | 32 | なし | 3.4548 | 2.7665 | 1.25× |
| 2,600,000 | 0 | なし | 12.9488 | 10.3301 | 1.25× |
| 2,600,000 | 1 | なし | 18.0874 | 15.3087 | 1.18× |
| 2,600,000 | 11 | なし | 25.8523 | 23.1976 | 1.11× |
| 2,600,000 | 32 | なし | 33.6232 | 31.6097 | 1.06× |
| 2,600 | 11 | あり | 0.1987 | 0.1767 | 1.12× |
| 2,600 | 32 | あり | 0.3165 | 0.2649 | 1.19× |
| 2,600 | 65 | あり | 0.8563 | 0.7520 | 1.14× |
| 26,000 | 11 | あり | 0.3258 | 0.3115 | 1.05× |
| 26,000 | 32 | あり | 0.6049 | 0.5874 | 1.03× |
| 26,000 | 65 | あり | 1.0789 | 1.0619 | 1.02× |
| 260,000 | 11 | あり | 3.9867 | 3.0369 | 1.31× |
| 260,000 | 32 | あり | 6.9598 | 5.9230 | 1.18× |
| 260,000 | 65 | あり | 7.4880 | 6.5800 | 1.14× |
| 2,600 | 33 | なし | 0.4085 | 0.3911 | 1.04× |
| 260,000 | 33 | なし | 5.1314 | 4.2057 | 1.22× |

### Native cold control

| sample数 | channel数 | 航法bit配列 | 旧 (ms) | 新 (ms) | 速度比 |
|---:|---:|:---:|---:|---:|---:|
| 2,600 | 12 | なし | 0.1725 | 0.1603 | 1.08× |
| 2,600 | 33 | あり | 0.3301 | 0.3316 | 1.00× |
| 260,000 | 12 | なし | 5.0069 | 4.9800 | 1.01× |
| 260,000 | 33 | あり | 8.7229 | 8.7818 | 0.99× |

### Public wrapper warm

| sample数 | channel数 | 航法bit配列 | 旧 (ms) | 新 (ms) | 速度比 |
|---:|---:|:---:|---:|---:|---:|
| 2,600 | 0 | なし | 0.2910 | 0.1524 | 1.91× |
| 2,600 | 12 | なし | 0.6370 | 0.5934 | 1.07× |
| 2,600 | 33 | なし | 1.4796 | 1.4459 | 1.02× |
| 26,000 | 0 | なし | 0.3407 | 0.3049 | 1.12× |
| 26,000 | 12 | なし | 0.8871 | 0.8697 | 1.02× |
| 26,000 | 33 | なし | 2.4525 | 2.3321 | 1.05× |
| 260,000 | 0 | なし | 3.5363 | 2.4989 | 1.42× |
| 260,000 | 12 | なし | 7.0985 | 5.9108 | 1.20× |
| 260,000 | 33 | なし | 3.5078 | 2.8927 | 1.21× |
| 2,600,000 | 0 | なし | 8.0105 | 6.7468 | 1.19× |
| 2,600,000 | 12 | なし | 11.9622 | 10.5129 | 1.14× |
| 2,600,000 | 33 | なし | 22.4956 | 20.1414 | 1.12× |
| 2,600 | 65 | あり | 0.9447 | 0.9000 | 1.05× |
| 260,000 | 65 | あり | 4.7680 | 4.3363 | 1.10× |

### Urban whole epoch / 12 visible rays

| sample数 | triangle数 | 旧 (ms) | 新 (ms) | 速度比 |
|---:|---:|---:|---:|---:|
| 2,600 | 12 | 0.7887 | 0.7728 | 1.02× |
| 26,000 | 12 | 0.5710 | 0.5715 | 1.00× |
| 260,000 | 12 | 2.4920 | 1.8720 | 1.33× |
| 2,600 | 1,200 | 0.5851 | 0.5616 | 1.04× |
| 26,000 | 1,200 | 0.6260 | 0.6255 | 1.00× |
| 260,000 | 1,200 | 2.5379 | 1.8947 | 1.34× |
| 2,600 | 12,000 | 1.1891 | 1.2012 | 0.99× |
| 26,000 | 12,000 | 1.2999 | 1.2931 | 1.01× |
| 260,000 | 12,000 | 3.3476 | 2.7244 | 1.23× |

1ms（2,600sample）のurban全体は0.99〜1.04×、26,000sampleもほぼ1.00×で、改善は小さい。
native cold controlは0.99〜1.08×で再利用の利得はなく、小さいpublic条件にも実行環境の変動が
ある。短いepochの速度向上や全GNSS/IMU pipeline/実データ位置精度の向上を主張しない。

**271ケースの旧版/reusable IQがbit一致**。うち既存263ケースでは新単発APIも旧版と一致した。
1〜260,000sampleのblock境界、0〜65channelのinline/pointer境界、全constellation、PRN境界、
無効PRN/system、航法bit/clamp/非unit bit、加算順/相殺、負phase/underflow、noise/seed境界を含む。
8件のgrowth/shrink/empty/入力変更の連続利用では以前の返却配列が変わらないことも確認した。
**全54測定条件の旧新版、全round/各回のIQ・urban metadataのhashも一致**。
信号/urban/reflection/diffraction/BVH/公開wrapperの243テストPASS。追加27テストのCUDA
memcheckは0 errors/0 bytes leaked、racecheckは0 hazards/0 errors/0 warnings。
新テストはclose/destructor、独立owner、入力・設定の再反映、invalid call後の再利用、返却値所有権、
release後の再作成、copy/deepcopy/pickle、旧extension fallbackとfresh noiseも検証する。

baselineを同じ構成でbuild・退避してから再現する。ローカルの退避済みbaselineを使う例:

```powershell
$env:PYTHONPATH = "python"
python benchmarks/bench_signal_workspace_compare.py `
  --baseline dist/signal_workspace_baseline/_gnss_gpu_signal_sim.cp312-win_amd64.pyd `
  --candidate build/diag/Release/_gnss_gpu_signal_sim.cp312-win_amd64.pyd `
  --baseline-dir dist/signal_workspace_baseline `
  --rounds 5 --repeats 20 --output dist/signal_workspace_recheck.json
```

source/extension/benchmark hashと全raw sampleはJSONに記録。
[比較スクリプト](bench_signal_workspace_compare.py)、
[検証テスト](../tests/test_signal_sim_workspace.py) を参照。

## Doppler KF geometry reuse (2026-10-12)

Doppler KFの残存コストを対象にし、100万粒子・12衛星の公開predict→PR→Doppler KF→
ESS判定/resample→estimate全体を **50.0132→42.7602 ms（1.17×、約14.5%短縮）** にした。
単体の12衛星KFは **34.5045→27.1058 ms（1.27×）**、32衛星KFは91.0176→71.3388 ms（1.28×）。
baselineは直前の共通共分散/候補先読み版で、過去の速度比を掛け合わせていない。

方向ベクトルと衛星速度を除いた観測値を、weighted meanの第一passとKF更新の第二passで
重複計算していた。位置/衛星はこの間変わらないため、**32衛星まで第一passのFP64値を保存して
第二passで再利用**する。距離のsqrtと方向の3 divisionを各有効粒子/衛星で二回→一回にする。
距離 < 1 mや無効weightでskipした値はvalid bitで区別し、未初期化cacheを読まない。
観測分散 `sigma² / weight` は64衛星までblock内で衛星ごとに一度計算する。
33衛星以上はgeometryを再計算し、65衛星以上は分散の追加shared領域も使わず、
従来の8×衛星数doubleのshared workspaceに戻す。従来の48 KiB境界（768衛星）もテストした。

weighted centering、衛星の順序、速度/共分散の更新式、symmetrization/floor/S guard、
log likelihood、FP64精度、既定resampler/API/設定は同じ。block sizeを128にする試行は
100万粒子で明確な追加利得がなく、従来の256を維持した。
所有者の追加cudaMalloc/workspaceはないが、**kernel local cacheは増える**。
このbuildのresource usageは16衛星版stack 512 bytes/thread、32衛星版1,024 bytes/thread
（各86 registers）、fallbackはstack 0（94 registers）、旧版はstack 0（92 registers）。
64衛星までshared使用量は8n→9n double。これらはコンパイル時resource量で、実行時peak memory
を測った値ではない。速度は3 warm-up後のsteady-stateで、初回のlocal-memory準備は測定外。

退避した5 source/extension hashを [直前の記録](pf_covariance_gpu_20261011.json) のcandidateと照合。
CUDA 12.8 / MSVC Release / compute capability 7.5 / C++20 / fast mathなし。
旧新版の実際のproduction extensionを別processで読み、5ラウンドで実行順を交互にした。
各条件3 warm-up後20回、最後のstream同期を含むwall time。表はround中央値の中央値。
初期化/状態復元・particle dump・出力確認は測定外。観測配列は事前作成する。

`KF` は1回のnative KF更新（入力staging/H2D/kernel/同期込み）。衛星は合成の軌道距離で、
非ゼロのsatellite velocity、観測noise、不均一な正のweightを使う。
`native PR + KF step` はRBPF covariance付きpredict→12衛星PR→12衛星KF→ESS取得→
ESS < 0.5NならB=60 coalesced resample→estimate取得→同期。
`public PR + KF step` は実際のPython wrapperのpredict/update(resample=False)/update_doppler_kf/
estimateで、PRとKFの両方のlog likelihoodを使う。入力検証/転送/ESS判定/推定値取得を含む。
`native PR control` は同じpredict/PR/ESS/条件付きresample/estimateでKFを呼ばない未変更経路。
全40条件を示す。

| 粒子数 | 条件 | 旧 (ms) | 新 (ms) | 速度比 |
|---:|:---|---:|---:|---:|
| 2,048 | KF / 衛星4 | 0.1507 | 0.1073 | 1.40× |
| 2,048 | KF / 衛星12 | 0.3467 | 0.2122 | 1.63× |
| 2,048 | KF / 衛星16 | 0.4334 | 0.2849 | 1.52× |
| 2,048 | KF / 衛星17 | 0.4381 | 0.2981 | 1.47× |
| 2,048 | KF / 衛星32 | 0.7774 | 0.4928 | 1.58× |
| 2,048 | KF / 衛星64 | 1.4799 | 1.1315 | 1.31× |
| 2,048 | KF / 衛星65 | 1.4910 | 1.4194 | 1.05× |
| 2,048 | native PR control | 0.3107 | 0.2669 | 1.16× |
| 2,048 | native PR + KF step | 0.5904 | 0.4968 | 1.19× |
| 2,048 | public PR + KF step | 0.7601 | 0.6851 | 1.11× |
| 10,000 | KF / 衛星4 | 0.2305 | 0.1784 | 1.29× |
| 10,000 | KF / 衛星12 | 0.5097 | 0.3985 | 1.28× |
| 10,000 | KF / 衛星16 | 0.6109 | 0.5045 | 1.21× |
| 10,000 | KF / 衛星17 | 0.6406 | 0.5382 | 1.19× |
| 10,000 | KF / 衛星32 | 1.1898 | 0.9842 | 1.21× |
| 10,000 | KF / 衛星64 | 2.3240 | 2.3329 | 1.00× |
| 10,000 | KF / 衛星65 | 2.3414 | 2.3981 | 0.98× |
| 10,000 | native PR control | 0.3978 | 0.3438 | 1.16× |
| 10,000 | native PR + KF step | 0.9257 | 0.7972 | 1.16× |
| 10,000 | public PR + KF step | 1.2341 | 0.8169 | 1.51× |
| 100,000 | KF / 衛星4 | 1.3622 | 1.0346 | 1.32× |
| 100,000 | KF / 衛星12 | 3.7359 | 2.9689 | 1.26× |
| 100,000 | KF / 衛星16 | 4.8675 | 3.8990 | 1.25× |
| 100,000 | KF / 衛星17 | 5.1543 | 4.1204 | 1.25× |
| 100,000 | KF / 衛星32 | 9.6334 | 7.5575 | 1.27× |
| 100,000 | KF / 衛星64 | 19.0541 | 18.4601 | 1.03× |
| 100,000 | KF / 衛星65 | 19.3612 | 19.4460 | 1.00× |
| 100,000 | native PR control | 1.4666 | 1.3667 | 1.07× |
| 100,000 | native PR + KF step | 5.1624 | 4.5365 | 1.14× |
| 100,000 | public PR + KF step | 5.2068 | 4.5156 | 1.15× |
| 1,000,000 | KF / 衛星4 | 11.9043 | 9.4477 | 1.26× |
| 1,000,000 | KF / 衛星12 | 34.5045 | 27.1058 | 1.27× |
| 1,000,000 | KF / 衛星16 | 45.8282 | 35.9567 | 1.27× |
| 1,000,000 | KF / 衛星17 | 48.6133 | 38.0960 | 1.28× |
| 1,000,000 | KF / 衛星32 | 91.0176 | 71.3388 | 1.28× |
| 1,000,000 | KF / 衛星64 | 181.4584 | 176.6603 | 1.03× |
| 1,000,000 | KF / 衛星65 | 184.2969 | 185.8269 | 0.99× |
| 1,000,000 | native PR control | 14.1337 | 14.3793 | 0.98× |
| 1,000,000 | native PR + KF step | 49.6287 | 42.5168 | 1.17× |
| 1,000,000 | public PR + KF step | 50.0132 | 42.7602 | 1.17× |

**1,566ケースの全16状態とlog weightsの出力hashが旧新版で一致**。
13粒子数（1/2、warp/block境界、8,193）×17衛星数（0〜128のcache/shared/growth境界）×7条件
（通常、zero/negative mask、距離0.5/1 m、不正S、全weight 0、sigma² underflow、NaN/Inf weight）
で二連続KFを実行した。13件の衛星数変更/観測scratch成長/resample混合利用と、
10万/100万粒子の12/32/65衛星6件を含む。全40測定入力の状態/log weights/ESS/返却推定値hashも
旧新版・全5ラウンドで一致した。NaN/Inf等は旧版の挙動を比較する条件であり、新しい精度保証ではない。

独立したCPU FP64のmatrix product/scalar Gaussian update参照に、速度/共分散/log likelihoodを
rtol/atol 2e-10で照合。mask/近接衛星/cache境界/48 KiB shared境界/連続利用/入力・出力所有権、
既存PF統計/wrapper/smoother/SMC/IMUを含む **214テストPASS**。
追加34テストのCUDA memcheckは **0 errors / 0 bytes leaked in 0 allocations**、
racecheckも **0 hazards / 0 errors / 0 warnings**。

100万粒子の未変更PR controlは14.1337→14.3793 ms（0.98×）でほぼ同じ。
64/65衛星のKFは1.03×/0.99×で利得が小さい。小粒子数の未変更controlは1.07〜1.16×に変動し、
小Nの大きい速度比をkernel変更だけの利得とはしない。非専有hostのwall timeであり、全条件の
改善、実データの位置精度向上、whole GNSS/IMUの速度比は主張しない。
今回の変更は以前に記録したsystematic CDF境界の変動も解消しない。

全sample/round中央値・出力/source/binary hashは [測定JSON](pf_doppler_gpu_20261012.json)。
再現用 [比較スクリプト](bench_pf_doppler_compare.py) はproduction extensionを実行する。
baselineは変更前に退避し、同じbuild設定でcandidateをrebuildする。

```powershell
$env:PYTHONPATH='python'
python benchmarks/bench_pf_doppler_compare.py `
  --baseline dist/pf_doppler_baseline/_gnss_gpu_pf_device.cp312-win_amd64.pyd `
  --candidate build/diag/Release/_gnss_gpu_pf_device.cp312-win_amd64.pyd `
  --baseline-dir dist/pf_doppler_baseline `
  --sizes 2048 10000 100000 1000000 --rounds 5 --repeats 20 `
  --build-configuration 'CUDA 12.8 / MSVC Release / compute capability 7.5 / C++20 / no fast math' `
  --output dist/pf_doppler_final_20261012.json
```

## Default PF resampling: common covariance and proposal prefetch (2026-10-11)

既定の `ParticleFilterDevice`（coalesced Megopolis、B=60）の更新全体を対象にした。
100万粒子の公開APIによるpredict→update→estimateは **14.2177→12.6016 ms（1.13×、約11.4%短縮）**。
同じ尤度更新後の再サンプリングは **7.1227→5.6140 ms（1.27×）**。
直前のsystematic重み準備版をbaselineにし、以前の速度比を掛け合わせていない。

初期化した速度共分散は全粒子で同じで、predictで加える共通process noiseでも同じまま。
この事実が保証される間、coalesced gatherは共分散の9成分をコピーせず、現在の配列を保持する。
100万粒子では一回のgatherでread/write合計144,000,000 bytesの論理的コピーを除く。
追加device bufferなし。個別状態のimportとDoppler KF更新では保証を無効にし、従来の全成分copy/
pointer swapに戻す。再初期化で保証を回復し、systematic/legacyとの混合利用も維持する。

候補のアドレスは受理済み祖先に依存しないため、Philoxの4回分の乱数と候補重みを先に読み、
受理判定を元の順序で行う。4で割り切れない末尾は元のscalar drawで処理する。
二つのsubsequence、seed、FP64 log weights/差分、既存のFP32 draw/log受理式を維持する。
候補indexの二回目の剰余は同値の条件付きwrapに替えた。浮動小数点の精度・受理式・公開APIは同じ。
共通乱数のwarp共有のみの案は速度差がほぼなく、採用しなかった。

退避した5 sourceとextension hashを [直前の記録](pf_systematic_gpu_20261011.json) のcandidateと照合。
CUDA 12.8 / MSVC Release / compute capability 7.5 / C++20 / fast mathなし。
旧新版の実際のextensionを別worker processで読み、5ラウンドで実行順を交互にした。
各条件3 warm-up後20回、各操作の最後のstream同期を含むwall timeで、表はround中央値の中央値。
初期化・状態/重み復元・出力検査は測定外。観測配列は事前作成する。

`normal` はlog weight σ=1.5、`共通cov` は初期化の保証あり、`個別cov` は粒子ごとの異なる共分散をimport。
B=0は反復なしのcopy診断で、公開設定の既定値はB=60。
`尤度後` は実際にpredictと12衛星のpseudorange weightを行った状態。
`native default step` はpredict→weight→ESS取得→ESS < 0.5NならB=60 resample→estimate取得→同期。
`public default step` は実際のPython wrapperのpredict/update/estimate（入力検証・転送・ESS判定込み）。
`Doppler KF step` はnative defaultに12衛星のKF updateを追加し、個別covのcopyへ戻る経路。
`systematic control` は同じnative stepで未変更のsystematicを使う。全60条件を示す。

| 粒子数 | 条件 | 旧 (ms) | 新 (ms) | 速度比 |
|---:|:---|---:|---:|---:|
| 2,048 | normal B0 共通cov | 0.0585 | 0.0640 | 0.91× |
| 2,048 | normal B60 共通cov | 0.0592 | 0.0794 | 0.75× |
| 2,048 | normal B240 共通cov | 0.1223 | 0.1157 | 1.06× |
| 2,048 | normal B0 個別cov | 0.0616 | 0.0417 | 1.48× |
| 2,048 | normal B60 個別cov | 0.0577 | 0.0661 | 0.87× |
| 2,048 | normal B240 個別cov | 0.1306 | 0.0673 | 1.94× |
| 2,048 | predict control | 0.0697 | 0.0526 | 1.32× |
| 2,048 | 12衛星 weight control | 0.1174 | 0.1100 | 1.07× |
| 2,048 | 尤度後 ESS control | 0.0476 | 0.0454 | 1.05× |
| 2,048 | 尤度後 resample B60 | 0.0519 | 0.0567 | 0.92× |
| 2,048 | resample後 estimate control | 0.0728 | 0.0655 | 1.11× |
| 2,048 | native default step | 0.3597 | 0.3353 | 1.07× |
| 2,048 | native Doppler KF step | 0.6522 | 0.5629 | 1.16× |
| 2,048 | native systematic control | 0.4403 | 0.4225 | 1.04× |
| 2,048 | public default step | 0.4531 | 0.4175 | 1.09× |
| 10,000 | normal B0 共通cov | 0.0788 | 0.0447 | 1.76× |
| 10,000 | normal B60 共通cov | 0.0733 | 0.0628 | 1.17× |
| 10,000 | normal B240 共通cov | 0.1746 | 0.1255 | 1.39× |
| 10,000 | normal B0 個別cov | 0.0648 | 0.0880 | 0.74× |
| 10,000 | normal B60 個別cov | 0.0762 | 0.0681 | 1.12× |
| 10,000 | normal B240 個別cov | 0.1897 | 0.1417 | 1.34× |
| 10,000 | predict control | 0.0786 | 0.0815 | 0.96× |
| 10,000 | 12衛星 weight control | 0.1416 | 0.1649 | 0.86× |
| 10,000 | 尤度後 ESS control | 0.0654 | 0.0627 | 1.04× |
| 10,000 | 尤度後 resample B60 | 0.0733 | 0.0529 | 1.39× |
| 10,000 | resample後 estimate control | 0.0763 | 0.0649 | 1.17× |
| 10,000 | native default step | 0.4058 | 0.3730 | 1.09× |
| 10,000 | native Doppler KF step | 1.0263 | 0.9384 | 1.09× |
| 10,000 | native systematic control | 0.5367 | 0.5075 | 1.06× |
| 10,000 | public default step | 0.5263 | 0.5273 | 1.00× |
| 100,000 | normal B0 共通cov | 0.1561 | 0.0835 | 1.87× |
| 100,000 | normal B60 共通cov | 0.4532 | 0.3270 | 1.39× |
| 100,000 | normal B240 共通cov | 1.1455 | 0.9395 | 1.22× |
| 100,000 | normal B0 個別cov | 0.1618 | 0.1580 | 1.02× |
| 100,000 | normal B60 個別cov | 0.4599 | 0.4297 | 1.07× |
| 100,000 | normal B240 個別cov | 1.1193 | 1.0274 | 1.09× |
| 100,000 | predict control | 0.3085 | 0.2866 | 1.08× |
| 100,000 | 12衛星 weight control | 0.4787 | 0.4349 | 1.10× |
| 100,000 | 尤度後 ESS control | 0.1486 | 0.1128 | 1.32× |
| 100,000 | 尤度後 resample B60 | 0.4550 | 0.2872 | 1.58× |
| 100,000 | resample後 estimate control | 0.1541 | 0.1451 | 1.06× |
| 100,000 | native default step | 1.5770 | 1.1770 | 1.34× |
| 100,000 | native Doppler KF step | 5.3885 | 5.0796 | 1.06× |
| 100,000 | native systematic control | 1.2128 | 1.1834 | 1.02× |
| 100,000 | public default step | 1.3980 | 1.1823 | 1.18× |
| 1,000,000 | normal B0 共通cov | 1.1264 | 0.5713 | 1.97× |
| 1,000,000 | normal B60 共通cov | 4.8146 | 3.6636 | 1.31× |
| 1,000,000 | normal B240 共通cov | 11.2378 | 8.5924 | 1.31× |
| 1,000,000 | normal B0 個別cov | 1.1300 | 1.1605 | 0.97× |
| 1,000,000 | normal B60 個別cov | 4.8465 | 4.3369 | 1.12× |
| 1,000,000 | normal B240 個別cov | 11.0133 | 9.1089 | 1.21× |
| 1,000,000 | predict control | 2.2603 | 2.2644 | 1.00× |
| 1,000,000 | 12衛星 weight control | 3.5974 | 3.5987 | 1.00× |
| 1,000,000 | 尤度後 ESS control | 0.5713 | 0.5637 | 1.01× |
| 1,000,000 | 尤度後 resample B60 | 7.1227 | 5.6140 | 1.27× |
| 1,000,000 | resample後 estimate control | 0.6954 | 0.6710 | 1.04× |
| 1,000,000 | native default step | 14.1212 | 12.5580 | 1.12× |
| 1,000,000 | native Doppler KF step | 48.8948 | 48.2415 | 1.01× |
| 1,000,000 | native systematic control | 8.9316 | 8.8704 | 1.01× |
| 1,000,000 | public default step | 14.2177 | 12.6016 | 1.13× |

**1,659ケースの全16状態/重み/推定値の出力hashが旧新版で一致**。
従来の1,274ケース（粒子数1〜100万、uniform/normal/dominant、seed 0/1/42/uint64最大、
B=0/1/15/60/240/257、端数warp/block、3 epoch、混合resampler）に385ケースを追加した。
追加条件はB=2〜9、微小差・大きなcommon shift・±Inf/NaN・極端値の重み、signed-zero covariance、
同じ初期状態の共通cov経路とimportで強制した全copy経路の比較、共通KF predictとDoppler KFの
無効化を含む。全60測定入力の最終状態/重み/返却値hashも旧新版・全5ラウンドで一致した。
今回の検証範囲の同値性であり、以前に記録したsystematicのCDF境界変動を解消したとはしない。

共分散保持・import/KF後の全成分copy・再初期化・混合利用・乱数の4回単位と末尾・既存の統計的
収束・wrapper/smoother/SMC/IMUを含む **180テストPASS**。新しい76テストのCUDA memcheckは
**0 errors / 0 bytes leaked in 0 allocations**。
100万粒子の未変更predict/weight/systematic controlはほぼ同じ。一方、Doppler KF stepは
48.8948→48.2415 ms（1.01×）で利得が小さい。小粒子数には遅い条件と未変更controlの変動があり、
特に10,000粒子の公開stepはほぼ同じ。非専有hostのwall timeであり、全条件の改善や
実データ位置精度・whole GNSS/IMUの利得を主張しない。次に大きい残存コストはDoppler KFの計算。

各サンプル・round中央値・出力/source/binary hashは [測定JSON](pf_covariance_gpu_20261011.json)。
再現用 [比較スクリプト](bench_pf_covariance_compare.py) は実際のproduction extensionを実行する。
baselineは変更前に退避し、header変更も含め同じ設定でcandidateをrebuildする。

```powershell
$env:PYTHONPATH='python'
python benchmarks/bench_pf_covariance_compare.py `
  --baseline dist/pf_warp_baseline/_gnss_gpu_pf_device.cp312-win_amd64.pyd `
  --candidate build/diag/Release/_gnss_gpu_pf_device.cp312-win_amd64.pyd `
  --baseline-dir dist/pf_warp_baseline `
  --sizes 2048 10000 100000 1000000 --rounds 5 --repeats 20 `
  --build-configuration 'CUDA 12.8 / MSVC Release / compute capability 7.5 / C++20 / no fast math' `
  --output dist/pf_covariance_final_20261011.json
```

## Systematic PF weight preparation (2026-10-11)

`ParticleFilterDevice(resampling="systematic")` の重み準備を効率化した。
最大log weightを求めた後の `exp(lw - max)` とブロック内合計を一つのkernelにまとめ、
保存したFP64重みを正規化で再利用する。従来は指数計算を正規化でも再実行していた。
粒子数Nに対して、指数計算を2N→N回、kernel起動を一回、global readをN double分減らす。
ブロック内/最終合計の加算順、division、Thrust CDF、target/seedの定義、祖先探索・状態コピー
を維持する。端数ブロックでも全threadがshared-memory集計に参加する。
追加device bufferはなく、公開API・設定・既存のMegopolis kernelも同じ。

比較対象は**直前の共分散gather最適化版**。退避したsource/extension hashを
[前のPF記録](pf_gather_gpu_20261011.json)のcandidate hashと照合してから実行した。
CUDA 12.8 / MSVC 2022 Release / compute capability 7.5 / fast mathなし、旧新版を別プロセスで
読み込み、ラウンドごとにbuildの実行順を交互にした。20回 × 5ラウンド、各条件3回warm-up。
表は各ラウンド中央値の中央値。各操作の終わりのCUDA stream同期を含むwall time。
測定前の状態/重み復元、particle dump、確認用probeは除く。

`resample` は重み準備からCDF・祖先選択・全状態コピー・重みreset・同期まで。
uniform、normal（log weight σ=1.5）、dominant（1粒子が0、その他-1000）をseed 42で比較した。
`native step` はpredict→12衛星のweight→ESS取得→ESS < 0.5Nならresample→estimate取得→同期。
観測配列の生成は測定外。`public step` は実際のPython wrapperのpredict→update→estimate→同期で、
入力検証・観測転送・ESS判定・状態更新・推定値取得を含む。初期状態は各回同じseedで復元する。
`Megopolis control` は同じnative stepでB=60の既存coalesced Megopolisを使う。

| 粒子数 | 条件 | 旧 (ms) | 新 (ms) | 速度比 |
|---:|:---|---:|---:|---:|
| 2,048 | resample uniform | 0.1535 | 0.1701 | 0.90× |
| 2,048 | resample normal | 0.1759 | 0.1529 | 1.15× |
| 2,048 | resample dominant | 0.1892 | 0.1424 | 1.33× |
| 2,048 | native step | 0.3966 | 0.4431 | 0.89× |
| 2,048 | Megopolis control | 0.3011 | 0.3553 | 0.85× |
| 2,048 | public step | 0.5550 | 0.4671 | 1.19× |
| 10,000 | resample uniform | 0.1889 | 0.2061 | 0.92× |
| 10,000 | resample normal | 0.2130 | 0.1867 | 1.14× |
| 10,000 | resample dominant | 0.2162 | 0.2233 | 0.97× |
| 10,000 | native step | 0.4780 | 0.4045 | 1.18× |
| 10,000 | Megopolis control | 0.3719 | 0.3362 | 1.11× |
| 10,000 | public step | 0.5872 | 0.4749 | 1.24× |
| 100,000 | resample uniform | 0.4744 | 0.4441 | 1.07× |
| 100,000 | resample normal | 0.4145 | 0.4031 | 1.03× |
| 100,000 | resample dominant | 0.3727 | 0.3404 | 1.09× |
| 100,000 | native step | 1.4815 | 1.4617 | 1.01× |
| 100,000 | Megopolis control | 1.5397 | 1.5612 | 0.99× |
| 100,000 | public step | 1.4921 | 1.3922 | 1.07× |
| 1,000,000 | resample uniform | 3.2535 | 2.9719 | 1.09× |
| 1,000,000 | resample normal | 2.5514 | 2.3493 | 1.09× |
| 1,000,000 | resample dominant | 2.1238 | 1.8373 | 1.16× |
| 1,000,000 | native step | 9.1824 | 8.8376 | 1.04× |
| 1,000,000 | Megopolis control | 14.0589 | 14.0542 | 1.00× |
| 1,000,000 | public step | 9.0897 | 8.8628 | 1.03× |

精度確認では、実際の旧新版extensionが持つnormalized weights・最大値/合計値を
[read-only probe](probes/pf_systematic_probe.cpp)で取得した。probeはPF codeをリンク・置換せず、
既存handleを読み、通常の実行経路には入らない。**948ケースすべてでnormalized weightsと
最大値/合計値のuint64が一致**。粒子数1〜100万、warp/block/最終集計の境界、uniform/normal/
dominant/sparse、±1e6 shift、同値・underflow・-Inf入力、seed 0/1/42/999999/1000000/uint64最大値を
含む（大粒子数は5重み・3seed）。全16状態成分は取得した祖先インデックスの元配列とbit一致し、
reset後の重みも一致した。19粒子数の3 epoch/mixed resamplerも含む**986ケース中977ケースで
旧新版の状態/祖先/重み/推定値出力hashが一致**。全24測定入力の最終状態/結果hashも
旧新版・全ラウンドで一致した。

**残る9ケースは全状態のbit一致ではない**。大粒子数のuniform/sparse、seed 0/1000000で、
正規化重みと合計値が一致する一方、CDFと境界上の祖先選択が異なった。旧版自身を同じ入力で
8回繰り返すとCDFが最大6.66e-16変動し、100万粒子のuniformで祖先が最大158,006個変わった
（各祖先のindex差は最大1）。sparseではindex差は最大17。新版自身でも同様の変動がある。
旧版のsparse CDFには数ulpの小さい下降も観測され、単調CDFを前提にしたCPU searchと異なる
境界選択になる場合があった。未変更の並列scanの丸め/実行順に伴う制約と判断し、例外名・
旧新版の出力/CDF hash・各8回の自己比較・CDF下降をJSONに残した。
祖先/状態の完全なbit再現性を達成したとは主張しない。この変更は系統的resamplerの既存の
target/seed定義や統計的性質を修正するものでもない。

独立したdyadic確率のCPU祖先参照、端数block/underflow、連続resample・解放・保持済み出力、
既存resampling統計・wrapper・annealed SMC・smoother・IMU adapterを含む**104テストPASS**。
新しい38テストをCompute Sanitizer memcheckで実行し、**0 errors / 0 bytes leaked in 0 allocations**。
小粒子数には遅い条件があり、未変更のMegopolis controlにも0.85〜1.11×のwall-time変動があった。
非専有ホストの測定で、全条件の改善や小粒子数の利得がkernel変更だけに由来するとは主張しない。
100万粒子のpublic stepでは約2.5%短縮したが、全体への効果は重み準備単体より小さい。
実データの位置精度改善、以前の速度比の積、whole GNSS/IMU pipelineの高速化は主張しない。

各sample/round、source/binary/probe/benchmark/input/output hashes、境界例外は
[測定結果](pf_systematic_gpu_20261011.json)、再現は
[比較スクリプト](bench_pf_systematic_compare.py)に保存した。
変更前のgather版を同じ構成でビルドし、拡張と `pf_device.cu` / `pf_device.h` /
`_pf_device_bindings.cpp` / `pf_device_runtime.py` / `pf_device_config.py` を
`dist/pf_systematic_baseline/` に退避してから新版をビルドする。
確認用probeは[独立CMake project](probes/CMakeLists.txt)で作る。

```powershell
$taskPybindDir = python -m pybind11 --cmakedir
$taskPythonExe = python -c "import sys; print(sys.executable)"
cmake -S benchmarks/probes -B build/pf_probe -G "Visual Studio 17 2022" -A x64 `
  "-Dpybind11_DIR=$taskPybindDir" "-DPython3_EXECUTABLE=$taskPythonExe" -DPYBIND11_FINDPYTHON=ON
cmake --build build/pf_probe --config Release --parallel 4
$env:PYTHONPATH = "python"
python benchmarks/bench_pf_systematic_compare.py `
  --baseline dist/pf_systematic_baseline/_gnss_gpu_pf_device.cp312-win_amd64.pyd `
  --baseline-dir dist/pf_systematic_baseline `
  --candidate python/gnss_gpu/_gnss_gpu_pf_device.cp312-win_amd64.pyd `
  --probe build/pf_probe/Release/_pf_systematic_probe.cp312-win_amd64.pyd `
  --build-configuration "CUDA 12.8; MSVC 2022 Release; compute capability 7.5; no fast math" `
  --rounds 5 --repeats 20 --output dist/pf_systematic_compare.json
```

## Combined resident LOS and reflection queries (2026-10-11)

LOS判定と一次反射を続けて行う場合、常駐版でも同じrx/satelliteを二度転送し、各々で
GPU同期していた。`BVHAccelerator.trace` / `trace_batch` は座標転送を一回にまとめ、
同じstreamで従来のLOS・反射kernelを実行してから一回同期する。必要な確保と初回の
反射訪問順転送は両kernelの起動前に済ませる。返却済み配列は後の呼び出しに影響されない。
FP64幾何・閾値・同値候補順・stack guard・並列/直列の選択とkernelの計算式は同じ。
`UrbanSignalSimulator.compute_epoch` は通常の一次反射経路でこのAPIを使う。
詳細な `compute_reflection_paths` を選ぶ設定では既存経路を使い、combined APIのない
building modelも従来の二つの呼び出しで動作する。不可視衛星は従来どおりray対象外。

比較対象は**直前の常駐BVH版**。退避した4 source/extension hashを
[常駐版の公開記録](bvh_resident_gpu_20261011.json)のcandidate hashと照合してから実行した。
CUDA 12.8 / MSVC 2022 Release / compute capability 7.5 / fast mathなし、同一プロセスで
旧新版の順を毎回交互にし、20回 × 5ラウンド、各条件3回warm-up。表は各ラウンド中央値の
中央値。native時間は入力検証・出力確保・座標転送・kernel・同期・全出力の取得を含む。
CPU BVH構築とPython class wrapperは除く。warmはsceneを再利用、coldはscene作成から
LOS+反射取得・closeまでを含む。singleは単一epochのLOS kernel、batchはbatch LOS kernel。

| 三角形数 | ray数 | LOS+反射の条件 | 旧 (ms) | 新 (ms) | 速度比 |
|---:|---:|:---|---:|---:|---:|
| 12 | 12 | warm batch | 0.1453 | 0.1257 | 1.16× |
| 12 | 12 | cold batch | 0.2182 | 0.1838 | 1.19× |
| 12 | 12 | warm single | 0.1648 | 0.1342 | 1.23× |
| 12 | 192 | warm batch | 0.2084 | 0.1872 | 1.11× |
| 12 | 192 | cold batch | 0.2770 | 0.2429 | 1.14× |
| 1,200 | 12 | warm batch | 0.1598 | 0.1311 | 1.22× |
| 1,200 | 12 | cold batch | 0.3953 | 0.3474 | 1.14× |
| 1,200 | 12 | warm single | 0.1808 | 0.1426 | **1.27×** |
| 1,200 | 192 | warm batch | 0.7262 | 0.6891 | 1.05× |
| 1,200 | 192 | cold batch | 0.8539 | 0.8214 | 1.04× |
| 12,000 | 12 | warm batch | 0.6248 | 0.5930 | 1.05× |
| 12,000 | 12 | cold batch | 1.9576 | 1.8643 | 1.05× |
| 12,000 | 12 | warm single | 0.6254 | 0.5961 | 1.05× |
| 12,000 | 192 | warm batch | 5.4408 | 5.4017 | 1.01× |
| 12,000 | 192 | cold batch | 6.7043 | 6.6222 | 1.01× |
| 1,200 | 4,097 | warm batch | 15.2184 | 15.1392 | 1.01× |
| 1,200 | 4,097 | cold batch | 15.3201 | 15.2620 | 1.00× |
| 1,200 | 4,097 | warm single | 15.2533 | 15.1779 | 1.00× |

次の都市シミュレーション測定は、旧新版の実際のPython wrapper/callerを使い、
`compute_epoch` の可視判定・ray取得・チャネル作成・実際のCUDA IQ生成までを含む。
形状はECEFへ変換した合成box群、12衛星が可視、信号生成は両側とも直前の同じ版、
2.6 MHz / noise seed 37。旧版もGPU形状を常駐させ、両方warm利用。CPU BVH構築は除く。
1,200三角形はNLOS 4衛星・反射3衛星・15チャネル、12,000三角形はNLOS 6衛星・反射6衛星・
18チャネル。12三角形の条件は全LOS・反射なし・12チャネルだった。

| 三角形数 | IQサンプル数 | 旧 compute_epoch (ms) | 新 (ms) | 速度比 |
|---:|---:|---:|---:|---:|
| 12 | 2,600 (1 ms) | 0.4571 | 0.4265 | 1.07× |
| 12 | 26,000 (10 ms) | 0.5821 | 0.5173 | 1.13× |
| 1,200 | 2,600 (1 ms) | 0.5033 | 0.4637 | 1.09× |
| 1,200 | 26,000 (10 ms) | 0.5684 | 0.5299 | 1.07× |
| 12,000 | 2,600 (1 ms) | 1.0267 | 0.9626 | 1.07× |
| 12,000 | 26,000 (10 ms) | 1.1870 | 1.1311 | 1.05× |

100合成ケース＋7 native測定入力でLOSが一致し、反射点/遅延のuint64が一致した。
うち有限入力の105ケースはsingle APIも旧版の別々の呼び出しと一致した。
同値反射・ECEF・退化・NaN/Inf埋め草・深い木・共有/重複leaf・大小batchの再利用に加え、
入力変更後のsnapshotと取得済み配列の非変更を検査した。6都市測定入力はIQのbyte列と
全結果配列が一致し、チャネル等のmetadataも一致した。不可視衛星を含むIQの比較、
詳細反射/回折・coverage・PF3Dとの互換性を含む**273テストPASS、1既存skip**。
最終版の11テスト（拡張/縮小/混合呼び出し、非連続/padding、解放/更新、不正shape、IQ）を
Compute Sanitizer memcheckで実行し、**0 errors / 0 bytes leaked in 0 allocations**。
7測定入力で同じquery列を実行した後のownerの `cudaMalloc` 要求量は旧新版で同じ。
新しいdevice bufferは追加していない。この確保量は全runtimeの実測peak VRAMではない。

これは重複する転送・同期の削減で、計算が重い大batchでは速度比1.00〜1.01×と差が小さい。
ホストは別の長時間計算と共有し、wall timeは変動する。以前の速度比との積、実データの
位置精度改善、whole GNSS/IMU pipelineの高速化はこの測定から主張しない。
各call/roundとsource/binary/benchmark/input/output hashesは
[測定結果](bvh_combined_gpu_20261011.json)、再現は
[比較スクリプト](bench_bvh_combined_compare.py)に保存した。
変更前の常駐版を同じ構成でビルドし、拡張と `bvh.cu` / `bvh.h` / `_bvh_bindings.cpp` /
`bvh.py` / `urban_signal_sim.py` を `dist/bvh_combined_baseline/` に退避してから新版をビルドする。

```powershell
$env:PYTHONPATH = "python"
python benchmarks/bench_bvh_combined_compare.py `
  --baseline dist/bvh_combined_baseline/_bvh.cp312-win_amd64.pyd `
  --baseline-dir dist/bvh_combined_baseline `
  --candidate python/gnss_gpu/_bvh.cp312-win_amd64.pyd `
  --build-configuration "CUDA 12.8; MSVC 2022 Release; compute capability 7.5; no fast math" `
  --rounds 5 --repeats 20 --output dist/bvh_combined_compare.json
```

公開wrapperの利用例（native bindingの返却順は `los, points, delays`）:

```python
from gnss_gpu.bvh import BVHAccelerator

bvh = BVHAccelerator(triangles)
los, delays, points = bvh.trace(rx, sat)  # rx[3], sat[n_sat, 3]; finite inputs
los, delays, points = bvh.trace_batch(rx_batch, sat_batch)  # NaN padding supported
bvh.release_gpu()
```

## Resident BVH geometry and work buffers (2026-10-11)

同じ建物形状を使う反復呼び出しで、BVH/三角形の検査・unpack・GPU確保/転送を毎回
行っていた。`BVHAccelerator` は初回のray呼び出しで `DeviceBVH` を作り、形状と作業
bufferを再利用する。rx/satelliteは毎回転送し、GPU同期・新しい出力配列への取得を行う。
入力/出力bufferは用途ごとの最大要求サイズまで保持するため、後の小さいbatchで再確保しない。
反射のCPU訪問順は作成時に1回求め、そのGPU確保/転送は初回の並列反射まで遅らせる。
512三角形/4,096 rayの経路選択、FP64幾何、同値候補順、stack guardとkernelは前版と同じ。
既存のstateless native関数も利用できる。

形状は所有するread-only snapshotになり、入力配列の後の変更に影響されない。
`update_triangles(new_mesh)` は新しいCPU BVHの構築成功後に古いGPU領域を解放し、
次回に新版を転送する。不正入力では現在の形状/sceneを維持する。
`release_gpu()` はCPU形状を保持してGPU領域を解放し、次回は再転送する。
pickle/deepcopyにはGPU handleを含めず、復元後は同じCPU形状から再転送する。
native ownerは作成時のCUDA deviceを記録して操作/解放し、元のdeviceへ戻す。
同一native ownerへの並行C++呼び出しは対象外。Python bindingはGILを保持する。

比較対象は**直前の並列反射最適化版**で、`c09dc38` の直列反射版ではない。
退避したsource/extension hashは[前の並列版の記録](bvh_multipath_gpu_20261011.json)と一致する。
CUDA 12.8 / MSVC 2022 Release / compute capability 7.5 / fast mathなし、同一プロセス、
旧新版の呼び出し順を毎回交互にし、20回 × 5ラウンド、各条件3回warm-up。
表は各ラウンドの中央値の中央値。native入力検証・変換・GPU転送・計算・同期・結果取得を含む。
旧版の呼び出しごとの確保/解放も含む。CPUでのBVH構築とPython class wrapperの検証は除く。
`warm` はsceneを再利用、`cold` はscene作成→呼び出し→closeまでを含む一回限りの利用。
LOS+反射は二つのnative呼び出しの連続実行で、各々のrx/satellite転送・同期を含む。

| 三角形数 | ray数 | 条件 | 旧 (ms) | 新 (ms) | 速度比 |
|---:|---:|:---|---:|---:|---:|
| 12 | 12 | warm LOS | 0.2345 | 0.1236 | 1.90× |
| 12 | 12 | warm 反射 | 0.3170 | 0.1991 | 1.59× |
| 12 | 12 | warm LOS+反射 | 0.5659 | 0.3202 | 1.77× |
| 12 | 12 | cold LOS | 0.2395 | 0.3002 | 0.80× |
| 12 | 12 | cold LOS+反射 | 0.5646 | 0.5053 | 1.12× |
| 12 | 192 | warm LOS | 0.2461 | 0.1293 | 1.90× |
| 12 | 192 | warm 反射 | 0.3880 | 0.2634 | 1.47× |
| 12 | 192 | warm LOS+反射 | 0.6387 | 0.3916 | 1.63× |
| 12 | 192 | cold LOS | 0.2044 | 0.2388 | 0.86× |
| 12 | 192 | cold LOS+反射 | 0.5921 | 0.5613 | 1.06× |
| 1,200 | 12 | warm LOS | 0.3126 | 0.1489 | 2.10× |
| 1,200 | 12 | warm 反射 | 0.3905 | 0.1846 | 2.12× |
| 1,200 | 12 | warm LOS+反射 | 0.7300 | 0.3439 | 2.12× |
| 1,200 | 12 | cold LOS | 0.3080 | 0.4286 | 0.72× |
| 1,200 | 12 | cold LOS+反射 | 0.7321 | 0.7298 | 1.00× |
| 1,200 | 192 | warm LOS | 0.3546 | 0.1916 | 1.85× |
| 1,200 | 192 | warm 反射 | 1.3071 | 1.0745 | 1.22× |
| 1,200 | 192 | warm LOS+反射 | 1.3092 | 0.8940 | 1.46× |
| 1,200 | 192 | cold LOS | 0.3579 | 0.4699 | 0.76× |
| 1,200 | 192 | cold LOS+反射 | 1.3392 | 1.3281 | 1.01× |
| 12,000 | 12 | warm LOS | 2.4148 | 0.3713 | 6.50× |
| 12,000 | 12 | warm 反射 | 2.4286 | 0.5332 | 4.56× |
| 12,000 | 12 | warm LOS+反射 | 4.8805 | 0.8216 | **5.94×** |
| 12,000 | 12 | cold LOS | 1.9323 | 2.4187 | 0.80× |
| 12,000 | 12 | cold LOS+反射 | 4.6620 | 3.4101 | 1.37× |
| 12,000 | 192 | warm LOS | 2.6492 | 0.6527 | 4.06× |
| 12,000 | 192 | warm 反射 | 7.9725 | 5.8796 | 1.36× |
| 12,000 | 192 | warm LOS+反射 | 9.8206 | 6.1868 | 1.59× |
| 12,000 | 192 | cold LOS | 2.1944 | 2.6922 | 0.82× |
| 12,000 | 192 | cold LOS+反射 | 9.5857 | 8.4014 | 1.14× |
| 1,200 | 4,097 | warm LOS | 0.4221 | 0.2864 | 1.47× |
| 1,200 | 4,097 | warm 反射 | 15.5065 | 15.2652 | 1.02× |
| 1,200 | 4,097 | warm LOS+反射 | 15.8698 | 15.5036 | 1.02× |
| 1,200 | 4,097 | cold LOS | 0.4562 | 0.5857 | 0.78× |
| 1,200 | 4,097 | cold LOS+反射 | 15.8935 | 15.8729 | 1.00× |

100合成ケースと全7測定入力で、反射点/遅延のuint64が旧版に一致した。
LOSは106入力で一致した（入力破棄後のsnapshotケースは反射だけを比較）。
前版のmesh/ray境界・ECEF・退化・非finite・同値反射・深い木・共有/重複leafに加えて、
同じsceneで大小batch・反射/LOS・並列/直列を交互に使い、入力変更/破棄後のsnapshotと
返却済み出力の非変更も検査した。形状更新、失敗時の維持、解放後の再利用、pickle/deepcopy、
coverage/candidate feature、PF3D互換性を含む155テストPASS、1既存テストskip。
最終ビルドの所有/拡張/解放/LOS-onlyを含む8テストをCompute Sanitizer memcheckで実行し、
**0 errors / 0 bytes leaked in 0 allocations**。新しいnative ownerは不正index/range/cycleを
GPU転送前に拒否し、構築失敗時にも確保済みbufferを解放する。

常駐分のメモリはscene解放/形状更新まで保持する。12,000三角形・12 rayでLOS+反射後に
ownerが持つ `cudaMalloc` 要求量は1,457,928 byte（約1.39 MiB）。全runtimeの実測peak VRAM
ではなく、所有する確保量。各条件の値はJSONの `resident_allocated_bytes` に保存した。
一回限りのcold LOSは0.72〜0.86×で遅くなるため、反復利用の高速化として扱う。
4,097 rayの反射は元の直列kernelを使い、時間の大部分を占める計算は変わらない。
ホストは別の長時間計算と共有しており、表のwall timeには変動がある。前版の速度比と
掛け合わせた推定、実データ精度の向上、whole GNSS/IMU pipelineの高速化は主張しない。

各call/roundとsource/extension/benchmark/input/output hashesは
[測定結果](bvh_resident_gpu_20261011.json)、再現は
[比較スクリプト](bench_bvh_resident_compare.py)に保存した。
変更前の並列版を同じ構成でビルドして拡張と `bvh.cu` / `bvh.h` / `_bvh_bindings.cpp` /
`bvh.py` を `dist/bvh_resident_baseline/` に退避し、新版を再ビルドする。

```powershell
python benchmarks/bench_bvh_resident_compare.py `
  --baseline dist/bvh_resident_baseline/_bvh.cp312-win_amd64.pyd `
  --baseline-dir dist/bvh_resident_baseline `
  --candidate build/diag/Release/_bvh.cp312-win_amd64.pyd `
  --build-configuration "CUDA 12.8, MSVC 2022 Release, compute capability 7.5, no fast math" `
  --rounds 5 --repeats 20 --output dist/bvh_resident_compare.json
```

公開wrapperの利用例:

```python
from gnss_gpu.bvh import BVHAccelerator

bvh = BVHAccelerator(triangles)
los = bvh.check_los_batch(rx, sat)
delays, points = bvh.compute_multipath_batch(rx, sat)
bvh.update_triangles(updated_triangles)
los = bvh.check_los_batch(rx, sat)
bvh.release_gpu()
```

## Fused signal simulation (2026-10-11)

衛星ごとのIQ生成と合成を1カーネルへ統合し、チャネル数 × サンプル数の中間IQ配列を
除いた。チャネルの入力順を維持し、各チャネルのfloat積を丸めてからfloat加算する。
`__fmul_rn` / `__fadd_rn`で、元の中間配列への保存で生じた丸めを残し、合成とのFMA化を
防ぐ。時刻・コード位相・搬送波位相のdouble計算、`__sincosf`、Philox、ノイズの加算式は同じ。
コード表と既存の信号モデル、外部APIは変更しない。

32有効チャネル以下は、このビルドで2,560 byteの設定をカーネル引数で渡し、設定専用の
GPU確保/転送も省く。CUDAの4 KiB引数上限内に収まることをcompile-timeで検査する。
33チャネル以上は設定bufferを確保/転送し、同じ式・加算順のカーネルを使う。
航法ビットは全有効チャネル分をまとめて1回確保/転送し、offsetで参照する。
空配列・未指定ではstatic bitを使い、配列末尾のclampも元と同じ。総bit数に比例した
host/device bufferを使う。無効system/PRNはスキップし、有効チャネル0では元のノイズ経路を使う。

32チャネル・2,600,000サンプルでは、IQ用の `cudaMalloc` 要求量は
**654.60 → 19.84 MiB**（中間634.77 MiBを削除）。コード表、航法ビット、CUDA runtime等を
含む実測peak VRAMではなく、ソースの確保サイズから計算した値。launchは有効チャネル数+1
から1になる。32チャネル以下・static bitではGPUの動的確保もIQ出力の1回だけになる。

旧版 `c09dc38` のsignal sourceと新版を同じ CUDA 12.8 / MSVC 2022 Release /
compute capability 7.5 / fast mathなしでビルドした。同一プロセスで旧新版の実行順を
毎回交互にし、20回 × 5ラウンド、各版3回warm-up。表は各ラウンドの中央値の中央値。
native入力検証/変換、GPU確保・転送・計算・同期・結果取得・解放を含む。
1回だけのコード表初期化とPython class wrapperの検証は除く。
sampling frequency 2.6 MHz、IF 0 Hz、noise -20 dB、seed 42の合成入力。

| サンプル数 | チャネル数 | 航法ビット配列 | 旧 (ms) | 新 (ms) | 速度比 |
|---:|---:|:---:|---:|---:|---:|
| 2,600 | 0 | なし | 0.0441 | 0.0442 | 1.00× |
| 2,600 | 1 | なし | 0.0598 | 0.0494 | 1.21× |
| 2,600 | 11 | なし | 0.1864 | 0.0824 | 2.26× |
| 2,600 | 32 | なし | 0.4471 | 0.1571 | 2.85× |
| 26,000 | 0 | なし | 0.0712 | 0.0713 | 1.00× |
| 26,000 | 1 | なし | 0.0917 | 0.0845 | 1.08× |
| 26,000 | 11 | なし | 0.6886 | 0.1639 | **4.20×** |
| 26,000 | 32 | なし | 1.2481 | 0.2974 | 4.20× |
| 260,000 | 0 | なし | 1.1088 | 1.1026 | 1.01× |
| 260,000 | 1 | なし | 1.5484 | 1.2765 | 1.21× |
| 260,000 | 11 | なし | 2.9076 | 1.4522 | 2.00× |
| 260,000 | 32 | なし | 5.9655 | 1.9179 | 3.11× |
| 2,600,000 | 0 | なし | 6.6289 | 6.7331 | 0.98× |
| 2,600,000 | 1 | なし | 7.7429 | 6.9439 | 1.12× |
| 2,600,000 | 11 | なし | 20.7390 | 9.1624 | 2.26× |
| 2,600,000 | 32 | なし | 46.2629 | 13.6752 | **3.38×** |
| 2,600 | 11 | 混在 | 0.2922 | 0.0920 | 3.18× |
| 2,600 | 32 | 混在 | 0.8349 | 0.1754 | 4.76× |
| 2,600 | 65 | 混在 | 1.6745 | 0.3104 | 5.40× |
| 26,000 | 11 | 混在 | 0.8610 | 0.2018 | 4.27× |
| 26,000 | 32 | 混在 | 1.5961 | 0.3383 | 4.72× |
| 26,000 | 65 | 混在 | 2.9917 | 0.6115 | 4.89× |
| 260,000 | 11 | 混在 | 2.9031 | 1.4716 | 1.97× |
| 260,000 | 32 | 混在 | 6.1608 | 2.1083 | 2.92× |
| 260,000 | 65 | 混在 | 11.3979 | 3.2061 | 3.56× |

263合成ケースで、全IQのfloat32をuint32として比較し旧新版がビット単位で一致した。
サンプル数1/2/255/256/257/2,600/26,000/260,000、チャネル数0/1/11/31/32/33/65、
5systemのPRN境界、コード位相境界/負の位相、IF、航法ビットの切替・clamp・空配列、
無効チャネル混在、相殺と加算順、符号付きゼロ/subnormal、固定seed 1/42/uint64最大値、
ノイズなし/低ノイズ/0 dBを含む。seed 0のビット比較はノイズがunderflowする条件だけで行う。
入力非変更も確認した。別の単独チャネルCUDA kernelとの合成比較、seedの再現性、seed 0の
fresh noise、信号捕捉、urban helper、E2E擬似距離/DLLを含む88テストがPASS。

ホストは別の長時間計算と共有しており、wall timeには変動がある。
有効チャネル0の経路は変更しておらず、その行の速度向上は主張しない。
既存シミュレーション出力を維持する最適化であり、実データの測位精度やwhole GNSS/IMU
pipelineの改善を示すものではない。各call/round、入力/出力hash、source/extension hashは
[測定結果](signal_fused_gpu_20261011.json)、再現は
[比較スクリプト](bench_signal_sim_compare.py)に保存した。
baselineはソース変更前に同じ構成でビルドして拡張/sourceを `dist/signal_speed_baseline/` に
退避し、新版を再ビルドする。ローカルの再現コマンド:

```powershell
python benchmarks/bench_signal_sim_compare.py `
  --baseline dist/signal_speed_baseline/_gnss_gpu_signal_sim.cp312-win_amd64.pyd `
  --baseline-source dist/signal_speed_baseline/signal_sim.cu `
  --candidate build/diag/Release/_gnss_gpu_signal_sim.cp312-win_amd64.pyd `
  --build-configuration "CUDA 12.8, MSVC 2022 Release, compute capability 7.5, no fast math" `
  --rounds 5 --repeats 20 --output dist/signal_fused_compare.json
```

## BVH multipath candidate evaluation (2026-10-11)

反射計算はrayごとのthreadが全BVH leafを順番に巡るため、少数rayではGPUの並列度が
足りなかった。512三角形以上、1〜4,096 rayの呼び出しを、三角形候補の並列評価と
最小delayの選択に変更した。倍精度の幾何計算、交差判定の閾値、元のBVH訪問順を維持する。
同値の正のdelayは訪問順が先の反射点を選ぶ。深い木の64-entry stack guardも元と同じ。
小さいmesh、4,096 ray超、重複訪問等でorderの確保量が増えるcustom BVHには元の経路を使う。

中間結果は三角形ごとに保存せず、最大32 tile/rayの勝者だけを保存する。
勝者bufferは最大4,096件（このビルドでは160 KiB）、別途、訪問三角形ごとにint32の
order bufferを使う。並列経路ではGPUへのBVH nodeアップロードを省き、候補と確定の
2カーネルで計算する。外部APIは同じ。

旧版 `c09dc38` のBVHソースと変更後を同じ CUDA 12.8 / MSVC 2022 Release /
compute capability 7.5でビルドし、同一プロセスで呼び出し順を毎回交互にして比較した。
20回 × 5ラウンド、各版3回warm-up。表は1epochの合成ray入力で、native呼び出しの
BVH unpack、GPU確保・転送、計算、同期、結果取得・解放を含むwall timeの中央値。
事前のCPU BVH構築とPython class wrapperの入力変換は測定外。

| 三角形数 | ray数 | 旧 (ms) | 新 (ms) | 速度比 |
|---:|---:|---:|---:|---:|
| 12 | 1 | 1.1220 | 1.1853 | 0.95× |
| 12 | 12 | 1.0562 | 1.0345 | 1.02× |
| 512 | 1 | 3.1935 | 1.1426 | 2.79× |
| 512 | 12 | 3.5624 | 1.3420 | 2.65× |
| 1,200 | 1 | 6.4988 | 1.5118 | 4.30× |
| 1,200 | 12 | 7.5815 | 2.0473 | 3.70× |
| 12,000 | 1 | 56.5891 | 3.8205 | **14.81×** |
| 12,000 | 12 | 64.4258 | 7.2077 | **8.94×** |
| 1,200 | 256 | 19.9481 | 4.1920 | 4.76× |
| 1,200 | 257 | 18.3079 | 3.2254 | 5.68× |
| 1,200 | 4,096 | 18.4866 | 14.5678 | 1.27× |
| 1,200 | 4,097 | 16.4718 | 16.3089 | 1.01× |

90合成ケースで、反射点とdelayの全doubleをuint64として比較し、旧新版がビット単位で
一致した（80ケースには非zero反射がある）。LOS出力と入力の非変更も照合した。
mesh/ray境界、複数epoch、ECEF移動、近距離/遠距離、座標scale、退化三角形、NaN/Inf、
深い木、共有/重複leaf、異なる反射点が同じdelayを持つ場面を含む。
BVH・raytrace・wrapper・urban multipathの60テストもPASS。実データの測位精度向上を
主張する変更ではなく、元の反射モデルの計算結果を維持した高速化である。

ホストは別の計算と共有しており、時間にはラウンド間の変動がある。
小さいmeshと4,096 ray超には並列評価を適用せず、その行の速度向上は主張しない。
raw時間、入力/出力hash、旧新版のソース/バイナリhashは
[測定結果](bvh_multipath_gpu_20261011.json)に保存した。
[比較スクリプト](bench_bvh_multipath_compare.py)の再現コマンド:

```powershell
python benchmarks/bench_bvh_multipath_compare.py `
  --baseline dist/bvh_speed_baseline/_bvh.cp312-win_amd64.pyd `
  --baseline-source dist/bvh_speed_baseline/bvh.cu `
  --candidate build/diag/Release/_bvh.cp312-win_amd64.pyd `
  --build-configuration "CUDA 12.8, MSVC 2022 Release, compute capability 7.5" `
  --rounds 5 --repeats 20 --output dist/bvh_multipath_compare.json
```

## Device PF covariance gather (2026-10-11)

既定の coalesced Megopolis（B=60）を段階別に測定した。100万粒子では
再サンプリング約11.6 msの大半が状態 gather にかかる。共分散9成分を
粒子ごとにストライド付きでコピーする代わりに、連続した成分を隣接スレッドへ
割り当てた。祖先インデックス、乱数列、反復回数、粒子数、倍精度の状態を維持し、
追加のGPU確保やカーネル起動はない。部分ブロックの余ったスレッドも、
有効粒子の共分散成分をコピーする。

変更前 `c09dc38` のPFソースと変更後を、同じ CUDA 12.8 / MSVC 2022 Release /
compute capability 7.5 でビルドした。ローカルGPUは6 GiBのNVIDIA CUDA GPU。
各条件20回 × 5ラウンド、各段階3回warm-up、旧版・新版の実行順を交互にし、
CUDA stream同期を含むwall timeの中央値で比較した。

| 粒子数 | B=60 resample 旧→新 (ms) | 速度比 | native pipeline 旧→新 (ms) | 速度比 |
|---:|---:|---:|---:|---:|
| 2,048 | 0.0546 → 0.0656 | 0.83× | 0.2812 → 0.3532 | 0.80× |
| 10,000 | 0.0732 → 0.0660 | 1.11× | 0.3220 → 0.3931 | 0.82× |
| 100,000 | 0.9057 → 0.3944 | 2.30× | 1.7694 → 1.3690 | 1.29× |
| 1,000,000 | 11.6016 → 4.8157 | **2.41×** | 21.7690 → 14.1663 | **1.54×** |

pipelineはnativeの `predict → weight（12衛星）→ ESS → resample → estimate`。
毎回B=60でリサンプリングする。観測アップロードと推定値取得を含み、
測定前の状態/重み復元とPython wrapperの入力検証は除く。
100万粒子のB=240も17.9188 → 11.4909 ms（1.56×）だった。

合成入力の1,274ケースで、全16状態成分とリセット後の重み、3epochのnative
pipeline、coalesced / legacy / systematicを交互に使う状態のSHA-256が旧新版で一致。
粒子数1〜100万、端数warp/block、反復0/1/15/60/240/257、uniform/normal/単一支配の
重み、seed 0/1/42/uint64最大値を組み合わせた（大粒子数はB=60、seed 1/42）。
CUDA統計/ラッパー/固定lagの41テストも通過し、8,192粒子でも重み分布への収束を
確認した。コピーだけの変更であり、実データの位置精度向上を主張するものではない。

ホストは別の長時間計算と共有している。小粒子数では変更しないpredict/weight等にも
ラウンド間の大きな変動があり、このwall time表ではpipelineの改善を確認できない。
全粒子数の速度向上は主張しない。先に試したブロック乱数の事前共有は100万粒子の
B=60で約1.01×に留まったため破棄し、乱数生成カーネルは変更していない。

各サンプル時間、出力hash、旧新版のソース/バイナリhashは
[測定結果](pf_gather_gpu_20261011.json)に保存した。旧新版の拡張を別プロセスで読み込む
[比較スクリプト](bench_pf_coalesced_compare.py)の再現コマンド:

```powershell
python benchmarks/bench_pf_coalesced_compare.py `
  --baseline dist/pf_speed_baseline/_gnss_gpu_pf_device.cp312-win_amd64.pyd `
  --baseline-source dist/pf_speed_baseline/pf_device.cu `
  --candidate build/diag/Release/_gnss_gpu_pf_device.cp312-win_amd64.pyd `
  --build-configuration "CUDA 12.8, MSVC 2022 Release, compute capability 7.5" `
  --rounds 5 --repeats 20 --output dist/pf_gather_compare.json
```

### PF測定時点のGPU最適化候補

同じローカル環境で別に測ったraytrace/signalの単独呼び出し（20回、3回warm-up）と
ソースの処理構造から優先順位を置いた。反射候補の並列化は上のBVH multipath節で
実装・比較済み。それ以外は未実装の候補で、速度向上の実績ではない。

| 対象 | 単独呼び出しの中央値 | 次に検証する箇所 |
|---|---:|---|
| multipath: 12,000三角形、1epoch、12衛星 | 約50.4 ms | 現在はrayごとのthreadが全leafを巡る。三角形候補の並列化と最小delay選択。同値候補の訪問順も維持する |
| LOS: 12,000三角形、1epoch、12衛星 | 約1.39 ms | 静的BVH/三角形の再アップロード・確保。1,000epochを一括処理すると約2.46 msなので、geometry常駐とbatch化を優先 |
| signal: 11GPS channel、2.6万 / 26万sample、seed=42のnoise | 約0.93 / 3.53 ms | channelごとの中間IQ配列とlaunch、nav bitの確保/コピー。融合時もfloatの加算順とseedを維持して比較 |

## CPU robust SPP (2026-10-11)

`robust_spp` の重み付き正規方程式で、衛星数 × 衛星数の対角行列を作る処理を
列ごとの重み掛けに変更し、設計行列と重み付き転置行列を反復間で再利用した。
倍精度、ロバスト重み、収束条件、有効衛星数の判定は維持する。

旧版 `c09dc38` と同じ合成観測を使い、各条件を100回 × 7ラウンド、旧版・新版の
実行順を交互にして比較した。以下はclean条件の1測位あたりの中央値。

| 衛星数 | 旧版 (µs) | 新版 (µs) | 速度比 |
|---|---:|---:|---:|
| 8 | 382.75 | 294.86 | 1.30× |
| 16 | 313.47 | 280.87 | 1.12× |
| 32 | 944.90 | 699.58 | 1.35× |
| 64 | 878.45 | 659.31 | 1.33× |
| 96 | 896.58 | 585.77 | 1.53× |

clean / Cauchy外れ値 / Huber外れ値の全15条件で、出力座標は旧版とbit単位で一致し、
入力配列も不変だった。速度比は1.08〜1.53×。別の回帰テストではゼロ重み、
有効衛星不足、特異な配置も比較し、53件が通過した。
衛星数96の対角行列72 KiBを反復ごとに確保する処理を削除した。これは削除した
行列のサイズであり、プロセス全体の最大メモリ使用量の実測値ではない。

この計測時には別の長時間検証が動いており、ホストは専有していない。
速度比は局所測定値であり、実データ全区間の精度、GNSS/IMU全体の遅延やGPU性能の
改善を示すものではない。各ラウンドの時間、出力差、ソースhashは
[測定結果](robust_spp_cpu_20261011.json)に保存した。再現用コマンド:

```bash
python benchmarks/bench_robust_spp.py --reference-ref c09dc38 --output dist/robust_spp_cpu.json
```

## GPU pipeline (2026-10-03)

この節は当時の実装の履歴。Device PFの15反復のlegacy Megopolisは、現在の
coalesced/B=60の精度・速度比較の基準には使わない（D-039）。

**GPU**: Turing 世代 consumer GPU, 6 GB VRAM (compute capability 7.5)
**CUDA**: 12.8 / Windows 11, MSVC 2022, Release build (`CMAKE_CUDA_ARCHITECTURES=75`)

| Module | Input Size | Time (ms) | Throughput |
|--------|-----------|-----------|------------|
| WLS Batch | 10K epochs | 2.95 | **3.39M epoch/s** |
| Particle Filter (host-buffered) | 1M particles | 106.45 | **9.39M part/s** |
| Particle Filter Device (predict+update) | 1M particles | 31.79 | **31.46M part/s** |
| Signal Acquisition | 32 PRN, 1ms | 135.00 | **237.0 PRN/s** |
| Vulnerability Map | 100x100 grid | 1.03 | **9.71M pts/s** |
| Ray Tracing | 1008 tri, 8 sats | 1.04 | **7.72M checks/s** |

- `Particle Filter` 行は毎回 `initialize → predict → update → estimate` を host 経由の
  バッファで実行する。`Particle Filter Device` 行はデバイス常駐の `predict → update`
  （ESS 判定とリサンプリングを含む）の定常ループ。同じ `predict → update` で比べると
  Device は host-buffered PF（約 85 ms）の **2.7–2.8 倍速**。
- 1M 粒子の Device 段階別内訳: predict 2.2 ms、weight 3.0 ms、ESS 0.5 ms、
  Megopolis resample 30 ms、estimate 0.6 ms。残りの律速はリサンプリング。

### Device Megopolis resampling (2026-10-03 fix)

このリサンプリングは以前、15 回の反復ごとに粒子状態（16 double/粒子：位置・速度・
速度共分散 9・clock）を丸ごとコピーしていた。そのため 1M 粒子で **225 ms** かかり、
Device PF は host-buffered PF より遅かった（153 ms、0.5 倍）。

受理判定はスロットの log-weight と乱数だけで決まり、粒子の中身には依存しない。
そこで反復を int32 の祖先インデックス配列だけで回し、最後に 1 回だけ状態を gather
するよう変更した。結果はビット単位で同一のまま、リサンプリングは 30 ms（7.4 倍）、
Device の `predict → update` は 153 → 32 ms（4.8 倍）になった。

## 2026-04-01 (historical)

**GPU**: Ada 世代 consumer GPU, 16 GB VRAM（型番は記録せず）
**CUDA**: 12.0

| Module | Input Size | Time (ms) | Throughput |
|--------|-----------|-----------|------------|
| WLS Batch | 10K epochs | 1.04 | **9.60M epoch/s** |
| Particle Filter | 1M particles | 81.44 | **12.28M part/s** |
| Signal Acquisition | 32 PRN, 1ms | 142.50 | **224.6 PRN/s** |
| Vulnerability Map | 100x100 grid | 0.62 | **16.14M pts/s** |
| Ray Tracing | 1008 tri, 8 sats | 0.71 | **11.32M checks/s** |

当時のボトルネック分析は、`cudaMalloc/cudaFree` を毎回呼ぶことだった。
`ParticleFilterDevice` で 5–10 ms まで改善する見込みとしていたが、上記のとおり、
実際の律速は Megopolis リサンプリングのコピー量だった。
