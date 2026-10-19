# 実走の処理区間を調べる (D245, 2026-10-16)

**今回は診断の追加で、速度改善の採用は0件。D236/製品/既定を維持する。** D244の小さな登録処理の置換が全PFで効かなかったため、PF内部のcall数と実際のRTK/GTSAM処理境界を調べた。詳細cProfileは実走で整合性と負荷に問題があり、全call treeを不採用とした。別方式として、関数トレースを使わない粗い区間記録を追加した。両方の生データ、異常、校正の不利な条件を保存する。

## 計測が計算を変えないことの確認

元D236のNativeFeedbackMassStepを呼び、粒子数・候補数/順序・prior・FP64・閾値・weight/choice/RNG・FIX/feedback/graphを変えない。新しい数値/CUDA/native拡張を作らず、D226 DLLとD233/D234/D235拡張を使う。製品側へのimportや既定dispatch変更はない。

RTKは診断用subclassから元のmethodを一度だけ呼ぶ。選んだGTSAM callだけを外部monitor経由にし、呼出し式のcallee/引数を一度ずつ元の順で評価する。変更した式を戻すと各methodのASTが一致する。__class__ closureを元と共有し、引数なしsuper()を保つ。module globalを書き換えるmethodや無関係なclosureは元functionへの粗い転送だけにする。元class/functionは編集せず、実走process内のclass aliasと元PF initializerは終了時に戻す。

field mainは凍結helperのASTを複製する。diagnostic aliasの導入と、元の保存state/deepcopy/encode/JSON/gzip/metadata処理への独立spanを除去すると、元mainのASTが一致する。original field helper/model/mathは不変。状態記録はPFタイマー外だがRTK process内なので、独立したdiagnostic_snapshotとして記録する。

29件の詳細診断testと24件の区間診断test、計53件PASS。各実行に同じ既知pytest設定warning1件。3backend×2seed×4構成×4epochで元D236の全保存state/RNG/graph/prior metadataと一致する。AST復元、C-call forwarding、super closure、例外時のspan/hook解放、既存profile hookの保護も確認した。今回GPU数値codeは変更せず、新しいCUDA sanitizer実行はない。過去のmemcheck/race/sync結果を新規PASSに数えない。

## 詳細call profileの失敗を保持

詳細方式はPF callの間だけcProfileを有効にし、getstats/export/pstats出力はPFタイマー外で行う。合成校正では各条件16callという外部回数に一致したが、実走では両設定とも外側の200callに対してrootが181call/recursive176と記録された。recursive扱いでself timeがcumulative timeを超える項目もあった。原因は未特定で、GTSAMやNumbaの特定処理に断定しない。**この全call treeと細かいself/cumulative比率は実走の律速根拠にしない。**

個別に独立metadataと照合できた値は区別して残す。_movesは各197valid block、addとepoch memo canonicalはcluster0が206522、cluster1が206341で元metadataと一致する。この一部一致や合成test PASSは、壊れたroot/tree全体の正しさを示さない。作業回数から速度改善を推定しない。

## PF単体の診断負荷の校正

各方式を元D236 CPU/診断方式/元CPU A/Aの3mode、3条件×5round×3groupで直列回転した。方式ごと135timed/9warm、計270timed/18warm。全warm保存stateと全timed state hash、実行中の従来metadataが元と一致した。PF全体に診断enable/disableまたはspan作成/終了を含め、restore/callback再束縛/hash/export/検証は時間外。1許可CPU/BLAS1/通常priority/非占有。

| 方式 | particles/unique/rows/groups | paired遅延率 | A/A遅延率 | PF負荷abs5%以内 |
|:---|:---|---:|---:|:---:|
| 詳細call | 64/6/64/4 | +118.52% | -0.77% | FAIL |
| 詳細call | 64/32/64/4 | +121.52% | +0.19% | FAIL |
| 詳細call | 128/128/128/8 | +87.91% | +4.16% | FAIL |
| 区間のみ | 64/6/64/4 | +6.17% | -2.09% | FAIL |
| 区間のみ | 64/32/64/4 | +3.49% | +3.60% | PASS |
| 区間のみ | 128/128/128/8 | +2.81% | -0.42% | PASS |

中央値はgroup→round→5roundの順で計算し、全roundを保存した。A/A abs5%基準は両方式3/3で通るが、個別roundの揺れは残る。詳細方式の負荷は約88〜122%で3/3が5%基準を超え、区間方式も1/3が超えた。区間方式の約2.8〜6.2%を正確な定常overheadと断定しない。これはPF単体の校正で、実走RTK subclass/GTSAM転送やserializationによる影響は校正していない。速度採用の試験ではなく、同じ方式の再測定はしない。

## 実走と粗い時間配分

Tokyo run3/seed20260710/400epoch、元cluster_relin0/1、native_cpuのみ。各方式×2設定×200paired PF stepで計800保存state行を、元D236の同設定CPU参照と比較した。軌跡NPZの全配列dtype/shape/bytes、全cfg、全既存metadataも一致。各prefix197valid/3skip。4prefix計1600 RTK processのwall/process-CPU区間を記録した。同じ2設定を別診断方式で観測した回数であり、800の異なる入力や複数route/seedを意味しない。今回は新しいGPU実走はない。

以下は**cProfileなしの区間方式だけ**で、phase_before=2の全200epochを集計する。workは記録したprocess時間から明示的snapshot spanを算術的に引いた値。排他的spanは重ならず、元の整数nsでworkへ正確に足し戻る。これはserializationがcache/schedulingに与えた影響を取り除いた無計測の実行時間ではない。比率は次に調べる範囲を選ぶ粗い観測値で、speedup予測や厳密な上限ではない。

| cluster | phase2 epoch | work秒 | snapshot秒 | PF排他 | phase2未分類 | AR排他 | 選択GTSAM排他 |
|:---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 200 | 16.000 | 25.992 | 53.7% | 23.6% | 9.9% | 5.8% |
| 1 | 200 | 19.144 | 32.108 | 48.7% | 26.9% | 10.5% | 6.1% |

PF内部はまだ大きい一方、_process_phase2の未分類部分も約24〜27%を占めた。この中にはprepare_double_difference_measurements、build_pim、ambiguity管理やmotion factor等がある。個々の内訳は今回計測しておらず、どれか一つを律速と断定しない。次はこの範囲を粗い境界で分け、全PF/RTKで効果の出る候補を選ぶ。全GTSAMをGPUに移せば大幅に速くなるという根拠はまだない。

選択GTSAM spanはupdate/calculateEstimate/Marginals/marginalCovarianceの一部だけで、全native/GTSAM実行を網羅しない。特にphase1のsuper先やmodel側のcallbackを一律には包んでいない。表の約6%を全FGO solverの割合と解釈しない。process-CPUは当processの他threadや量子化を含み、小さいspanのcycle計測ではない。

cluster0: phase1 AR p50=8.608ms、最大2回 ep2=17.335s, ep1=1.205s。
cluster1: phase1 AR p50=10.193ms、最大2回 ep2=0.307s, ep1=0.037s。
phase1合計だけを見るとARが大きく見えるが、最初のepochに偏りがある。起動時も除かず合計へ含め、p50/p95/max/遅い5回を別に保存した。起動直後の原因を今回の記録だけで特定しない。

## 保存と次の候補

全実走source/input/external tc/PE/DLLのhashと終了時alias復元を照合した。36製品source、5installed extension、PF build PE、数値DLLと3拡張、50既報、24archive/3264filesを保全する。既存codecのscalar NaN payload正規化を引き継ぎ、新しい許容差は追加しない。1route/seedのprefixだけで、全route/multiseedの測位精度・GPU実走・whole graph速度gateは未完了。

優先する次の調査は、phase2未分類の観測準備/IMU/motion factor境界。PFの候補構築とcanonical検査も残るが、D239/D240/D241/D244の不採用結果を実証済み改善として再利用しない。新しい実装は全粒子・全候補・元FP演算/RNGを維持した上で、完全PF/RTKの対照比較を通して判断する。診断コードの推奨入口は[cProfileなしの実走helper](../benchmarks/bench_rbpf_process_profile_field.py)。初回の詳細helperは不採用診断として記録を残す。

[全結果](../benchmarks/rbpf_call_profile_20261016.json)、[完了照合](../benchmarks/rbpf_call_profile_completion_check_20261016.json)、[区間校正](../benchmarks/bench_rbpf_process_profile.py)。
