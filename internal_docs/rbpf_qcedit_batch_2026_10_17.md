# 観測品質処理のNumba一括化 (D268, 2026-10-17)

品質処理全体は両clusterで約50.3%短縮しcomponent gate 2/2。実走の400保存PF状態・全軌跡・設定・従来metadataは元処理にbits一致した。全体prefix gate 0/1。製品/既定は維持する。

## 計算を保った変更

D246の粗い境界診断でqceditがphase2の約5〜6%だったため、衛星ごとのPython/Numba往復を1回のcompiled loopへまとめた。衛星順、除外/NaN/health/仰角/CNR/LLIの全判定、全観測・周波数、GFの逐次状態更新を保つ。geodist/satazel/_qc_signal_checks/_gf_slip_checkは元のNumba関数をそのまま呼ぶ。fastmath=False、FP64、元の位置予測式とsystem cache builder。閾値・粒子・候補・prior・weight/RNG・FIX/feedback/graphを変えない。GPU数値source/PE/転送は変更しないCPU処理の実験である。

Nav/Obs/通常array/layout/非重複/信号型/元のbinding・code・source・error policyを確認し、ログありや特殊入力は元qceditへ戻す。未知fork/APIへの一般互換wrapperではない。追加のthreshold/波長/除外表はcall内の一時配列で、状態や確率の跨epoch cacheはない。heap peakは測定していない。通常の変更されていないruntime定数/依存関数を前提とし、外部からの任意のメモリ・compiled-helper改変や同一状態への並行書込み、OOM時の途中状態は一般保証しない。

初版31tests、list/integer域を別pathで拡張した53testsを保存した。最初の実入力replayでは800call全てfallbackだったため速度測定はせず、実入力の普通の3float list、整数CNR閾値、信号の文字列表現cacheを確認した。最終版はguardの2式だけを変更し、戻すと元guard ASTに一致する。元の数値bodyは同一。元Nav/信号objectや値を書き換える型適合処理はしない。最終64tests PASS（既存pytest設定warning1）。全800実QC入力がfast loopを使い、全返却/edt/el/gf/slip/観測arrayのbitsが一致した。複数seed/周波数の8epoch、重複衛星、空入力、NaN payload/signed zero、閾値境界、GLONASS周波数、fallbackの例外/途中状態を含む。

## 全QCの制御計測

Tokyo run3/seed20260710、cluster0/1各400QC call（両受信機・全200phase2epoch）。reference/batch/A/Aを5round×3groupで順序回転し、36000timed/2400warmを保存した。全guard/list変換/metadata/table作成/allocation/compiled loop/返却をtimer内、restore/deepcopy/検証/hash/IOを外とした。許可CPU1本、BLAS1、通常優先度、非占有。各groupの400call整数ns合計のpaired短縮→3group中央値→5round中央値。20%以上、4/5改善、A/A絶対中央値5%未満が事前基準。

| cluster | QC全体短縮 | 改善round | A/A | gate |
|---|---:|---:|---:|---|
| 0 | +50.289% | 5/5 | -1.386% | PASS |
| 1 | +50.340% | 5/5 | -0.989% | PASS |

独立auditで全raw順序/件数/ns/全出力hashを照合し、別の元/候補800入力再生も全状態bits一致、fast800/fallback0。元の品質処理全体に対する結果であり、GPU化率やPF/FGO全体の短縮率ではない。完了計測の再実行はしない。

## 実走の精度と全RTK時間

両cluster各400RTKepoch/200保存PF状態を元D236と比較し、全state/RNG/全軌跡array dtype/shape/bytes・全cfg/従来metadataが一致した。候補QCはphase1込み計1600call、全fast/fallback0。phase2の800QC入力/出力も全値・array bits/layout・可変object参照関係が一致する。

初回auditはcluster1最初のpickle全byte比較で失敗した。元auditor/sourceを保全し、数値/metadata全182leafと可変参照関係には差がなく、immutable metadataのmemo表現だけが異なることを確認した。別checkerで全fieldを型・順序・float/array bytes・layout・可変aliasまで照合する。ULP/zero符号/キー順/flags/metadata型/aliasの6負例を検出。許容差やNaN正規化、field除外は追加していない。

全RTK速度helperの初回は辞書keywordの変換件数検査で、warm/timed epoch開始前に停止した。初回plan/log/sourceを保存し、別helperでkeyword visitorだけを追加、undo ASTを確認した。数値body・試験・基準は同一。

状態保存/JSON/gzip/metadata診断文だけを元field mainから外し、戻すと元ASTに一致する。元PF call、観測/IMU/FGO/ARと候補QCを全てprocess timerへ含める。3完全prefixをwarm、5round×3variantの15prefixを判定対象とし、全400epochと全200phase2を集計する。初期の遅いepochも残す。両集計ともpaired中央値3%以上、4/5改善、対応A/A絶対中央値5%未満を事前基準とした。全軌跡/設定/入力/source/PE/DLLを毎runで照合する。

| cluster | 集計 | 短縮 | 改善round | A/A |
|---|---|---:|---:|---:|
| 0 | phase2全200epoch | +2.987% | 3/5 | +1.335% |
| 0 | 全400epoch | +3.918% | 3/5 | +1.992% |

cluster0の全体gateがFAILだったため、事前の段階方針に従いcluster1の全体速度比較は行わない。 全18prefix、7200RTKepoch、3600PFcallの軌跡/設定一致を確認した。これは1route/seedのprefixで、application起動/I/O・全route・多seedやGPU backend全体の速度を示さない。全round/不利な対照をJSONに残す。

## 保存と入口

36製品source・5installed PE・PF build PE・96既報・47archive/8447filesを保全する。新しいCUDA sanitizerは実行していない。診断launcher2件は無出力で停止したため当該exec treeだけを終了し、job別Numba cache/watchdogで同sourceのguard診断を完了した（原因未特定、timed callの再試行ではない）。Ruff/compile/UTF8/EOL/diff/公開機種名/artifact policyとarchive hashを照合する。

[結果](../benchmarks/rbpf_qcedit_batch_20261017.json)、[完了照合](../benchmarks/rbpf_qcedit_batch_completion_check_20261017.json)、[最終factory](../benchmarks/rbpf_qcedit_batch_field_schema.py)、[全QC計測](../benchmarks/bench_rbpf_qcedit_batch.py)、[実走helper](../benchmarks/bench_rbpf_qcedit_batch_field_checked.py)、[全RTK helper](../benchmarks/bench_rbpf_qcedit_batch_route_checked.py)、[厳密capture checker](../benchmarks/rbpf_qcedit_capture_check.py)。元の[数値loop](../benchmarks/rbpf_qcedit_batch.py)、[list adapter](../benchmarks/rbpf_qcedit_batch_list.py)、[初期schema adapter](../benchmarks/rbpf_qcedit_batch_schema.py)、[元field helper](../benchmarks/bench_rbpf_qcedit_batch_field.py)、[事前検査で停止したroute helper](../benchmarks/bench_rbpf_qcedit_batch_route.py)も保持する。試験は[初版](../tests/test_rbpf_qcedit_batch.py)、[list](../tests/test_rbpf_qcedit_batch_list.py)、[integer](../tests/test_rbpf_qcedit_batch_schema.py)、[最終実入力](../tests/test_rbpf_qcedit_batch_field_schema.py)。
