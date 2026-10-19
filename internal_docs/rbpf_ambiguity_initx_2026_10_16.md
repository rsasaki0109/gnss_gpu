# 共分散の行・列初期化を一括代入する (D246, 2026-10-16)

**保護版の曖昧度更新全体は約62〜64%短縮。RTK全体の効果は未確認で、元処理どうしの対照が約5〜8%揺れたため採用しない。全体gate FAIL。** 製品/既定は変えず、D236のPF計算を維持する。初版は特殊な引数の例外時に途中状態が異なるため不採用とし、元source/測定/negative controlを保全した。

## 処理境界の再調査

D245で未分類だったphase2を、DD観測準備・衛星位置・qcedit・曖昧度管理・IMU積分・motion factor・残差処理へ分けた。元のspan/AST forwarding helperは別globalsで使い、元helperのmethod list/call map/globals/sourceを変えない。選択callを戻すと元method ASTが一致する。super closure、評価順、例外、整数wall/process-CPU区間を7testsで確認し、実走cluster0/1計400保存PF state・全軌跡array bits・全cfg/従来metadataがD236に一致した。関数単位cProfileは使わない。

| cluster | PF排他 | 曖昧度更新 | qcedit | 衛星位置 | IMU積分 | Doppler LS | phase2未分類 |
|:---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 52.56% | 8.75% | 5.33% | 4.94% | 0.32% | 0.74% | 1.05% |
| 1 | 48.61% | 9.55% | 5.90% | 5.55% | 0.39% | 0.79% | 1.16% |

全phase2の200epochを含み、明示snapshot spanだけを算術的に引いた計測内の比率。追加adapterの負荷は校正しておらず、無計測時間・速度上限・全GTSAMの割合ではない。IMU積分単体は小さく、曖昧度更新が約9%だったため、その内部を調べた。

## 変更と精度

元のinitxはx[i]を設定した後、range(nx)の各jについてP[j,i]とP[i,j]へ0、対角へv0を2回書く。候補は通常の非重複要素を持つC/F contiguous float64 Pで、元と同じx[i]を書き、P[:nx,i]とP[i,:nx]へ+0、対角へ元v0を代入する。浮動小数点演算を追加/並べ替えず、値の書き込みだけをまとめる。nx外のborderは触らない。元の曖昧度更新loop・呼出し/引数の順序・全観測・全粒子/候補/prior・FP64・閾値・weight/choice/RNG・FIX/feedback/graphはそのまま。読み込み済みPがゼロという仮定で処理を省略しない。

literalな元initx codeと外部source hashをconstructorで照合する。完全な元Nav schema、通常のndarray/dtype/shape/writeability/stride/添字、range binding、NumPy error policy等の条件外は元initializerへ戻す。constructor/source検証はprocessタイマー外。新しい確率/跨epoch cache、native/CUDA buffer、数値DLLやGPU kernelの変更はない。既存Pの行/列へ直接代入し、行列全体のコピーを作らない。新heap peakの測定はしていない。

初版31testsはC/F行列、符号付きゼロ/NaN、border、stride/dtype/read-only/範囲外/strict-policyの例外・途中状態、3seed×2NF×4epochの全曖昧度更新stateを確認した。しかし後からv0=10**400を調べると、対角でOverflowErrorになる前に初版が後続の行/列までゼロにしてしまい、元の途中Pと異なることが判明した。初版は実入力で一致していても一般の途中状態保護が不足しており、採用しない。

保護版はexactな通常scalar型と安全な整数幅を先に確認し、巨大Python整数・custom/wider scalar・特殊添字は、書き込み前に元関数へ戻す。整数を縮小して計算する変更ではない。元31test bodyを同じまま再利用し、初版のnegative control、巨大整数conversion errorの途中全配列、wider scalar fallback、明示NaN payloadを追加した38testsがPASSした。各test実行の既知pytest設定warning1を残す。正常Nav schema以外やOOM/同じ状態への並行書込みを一般保証するものではない。

## 実入力の完全な曖昧度更新比較

初版と保護版それぞれcluster0/1各200paired PF step、各400保存state行・全NPZ配列dtype/shape/bytes・全cfg・全従来metadataが元D236 CPUと一致した。各prefix197valid/3skip。両版の全400曖昧度更新captureも、入力/全変更配列/obs/signal metadata/呼出しcounterまで一致した。各prefix全体のinitxはcluster0で5628、cluster1で5626call、全てfast path、fallback0。

保存したphase2全200×2設定=400入力を元のupdate_ambiguitiesへ再生する。RTK componentのx/P/outc/edt/slipを全て保存・照合し、観測配列も不変を確認した。独立auditorは元/候補をもう一度非計測で全400caseへ適用し、全変更array bitsとinitxの全引数/順序のpickle bytesも一致させた。phase2 capture中の元initx呼出しは計5189。これは初版/保護版とも同じで、各版の18,000timed/1,200warmの全出力とcounterも検証した。

1許可CPU/BLAS1/通常priority/非占有、各版3mode×5round×3group×全400case。restore/deepcopy/IO/状態検証は時間外、曖昧度更新loop全体と候補guard/counter/一括代入を時間内とした。roundごと全200caseの時間を合算し、5roundのpaired短縮率中央値で判定する。20%以上、4/5以上改善、A/A abs5%未満がcomponent gate。初版の数値はsource固定のまま保持し、保護版は新しいsource versionとして別に測定した。同じsourceの完了済み計測を繰り返して採用へ近づけない。

| 実装 | cluster | 曖昧度更新全体の短縮 | 改善round | A/A短縮 | component gate |
|:---|:---:|---:|:---:|---:|:---:|
| 初版・不採用 | 0 | +63.73% | 5/5 | +1.13% | PASS |
| 初版・不採用 | 1 | +64.20% | 5/5 | +0.67% | PASS |
| 保護版 | 0 | +62.47% | 5/5 | +1.48% | PASS |
| 保護版 | 1 | +63.67% | 5/5 | -0.70% | PASS |

初版はcomponent速度基準を通ってもscalar conversion時の途中状態の問題で不採用。component結果をRTK全体の高速化率へ読み替えない。

## RTK process全体の制御比較

保護版だけを元D236 CPU、保護版、元CPU A/Aで直列回転する。元field mainから観測用boundary/state/deepcopy/encode/gzip/metadata文だけを外し、戻すと元ASTに一致する。PFの元call・戻り値・処理は残す。RTK processの入口から出口までを測り、元PF・観測/IMU/FGO/AR処理と新guardを全て含める。constructor/source検証・入力file decode・出力/状態保存は時間外。精度の詳細state比較は前述の別runで行い、計測runでは毎回全軌跡NPZと全cfg/source/input/DLL/PEを照合する。

Tokyo run3/seed20260710/400epoch、cluster0の1prefix。最初の3完全prefixは事前指定warmとして全て保存し、残り5round×3mode=15prefixを判定に使う。全18run、7200 RTK epoch/3600 PF callの軌跡array bitsと全cfgがD236と一致した。主基準は全200phase2 epoch時間合計のpaired中央値5%以上・4/5改善・A/A abs5%未満、さらに全400epochで3%以上・4/5改善・A/A abs5%未満。初期の遅いepochも各runの集計に残し、都合の良いepochを除かない。

| round | phase2短縮 | phase2 A/A | 全400epoch短縮 | 全400 A/A |
|:---:|---:|---:|---:|---:|
| 1 | +4.82% | -5.33% | +4.94% | -2.78% |
| 2 | +38.56% | +31.22% | +38.13% | +27.01% |
| 3 | -5.39% | -43.11% | +4.84% | -28.89% |
| 4 | +15.61% | +8.73% | +17.58% | +5.16% |
| 5 | -4.87% | +8.05% | -16.03% | +11.34% |

phase2中央値+4.82%（3/5）、A/A +8.05%。全400epoch中央値+4.94%（4/5）、A/A +5.16%。全体gate FAIL。phase2は中央値5%/4round基準にも届かず、A/Aはphase2・全400epochともabs5%未満の基準を超えた。全体の効果が未確認なので採用しない。全roundと不利な対照を保存し、再測定しない。これはRTK processのprefix比較で、全application起動/I/O、全route/multiseedやGPU backendの速度を示さない。cluster1はstate/replay精度のみで、全RTK速度の制御比較はしていない。

## 保存と利用範囲

最初のtest launcherは無出力で停止し、この作業のprocess treeだけを終了して記録を保存した。原因は未特定。source不変でjob別Numba cacheとwatchdog/bootstrapを加えた実行では31testsが通った。初回component launcherはthreadpoolctl import不足でplan/warm/timed開始前に停止。既存のpure Python moduleだけをprivate dependencyへhash付きで複製し、測定sourceは変えない。初版の特殊入力negative control、全旧source/結果、保護版の新source/結果も保持する。

36製品source、5installed PEs、PF build PE、D226 DLLと3拡張、52既報、25archive/3439filesを保全する。新しいCUDA sanitizerは実行していない。既存codecのscalar NaN payload正規化は継承し、array/storeのpayloadは別の明示testで照合した。数値許容差は追加しない。製品/既定は維持し、現時点の入口は[保護版factory](../benchmarks/rbpf_ambiguity_initx_checked.py)と[実走比較helper](../benchmarks/bench_rbpf_ambiguity_initx_checked_route.py)。元外部cssrlib forkの凍結sourceに限定した実験で、任意versionへの互換APIではない。

[全結果](../benchmarks/rbpf_ambiguity_initx_20261016.json)、[完了照合](../benchmarks/rbpf_ambiguity_initx_completion_check_20261016.json)、[実入力再生](../benchmarks/bench_rbpf_ambiguity_initx_checked.py)。
