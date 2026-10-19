# QCと曖昧度管理の厳密な組合せ (D269, 2026-10-17)

このnativeCPU/cluster0 prefixの事前速度gateはFAIL。全体の効果が未確定のため採用しない。 157tests PASS。CPU/GPU×cluster0/1の計800保存PF状態、全軌跡/設定/従来metadata、全QC/曖昧度captureが同backendの元処理にbits一致した。製品/既定は維持する。

## 新しい構成

D247の曖昧度管理（過去component約81%短縮、保護版initxを含む）とD268のQC（過去component約50%短縮）を同じRTK classへ接続する。両方の実装・数値helper・guard・fallback・ソースをそのまま使い、既存の単体全RTK gate FAILは保持する。これは新しい組合せの実験で、同じ候補の完了済み時間を取り直すものではない。

QCが作るedt/slip/gf、基準局/roverのnav切替、single difference、周波数ごとのslip消費とambiguity resetが連続するときも、元の値と順序を保つ。全観測/粒子/候補/prior、FP64の式/演算順、閾値、RNG、FIX/feedback/graphは変更しない。新しいcache・数値kernel・GPU転送やbufferは追加しない。factoryは両engineへの参照を持ち、元のinitx/update/qceditのidentityとmethod継承を確認する。

既存の曖昧度69testsとQC64testsを変更せず再利用し、新しい相互作用24caseを追加した157testsが最初の実行でPASS。3seed×nf1/2/3×base-list/roverの18case各8epochで、QC直後とambiguity直後のx/P/outc/edt/slip/el/gf/観測全bitsと信号metadataが一致する。残る6caseはログ/custom initxのcounter観測順/overflow/stride/strict-policy/QC binding変更。既存pytest設定warning1、元のint64 overflow負例warning1を保存した。許容差やNaN正規化は追加していない。

## 実データの接続確認

Tokyo run3/seed20260710、各400RTKepoch/200保存PF state。CPUは既存の凍結QC/ambiguity capture、GPUは元処理を新規captureして、どちらも既存D236の全PF保存state/trajectory/cfg/metadataと照合した。候補はCPU/GPU各cluster0/1の4prefix。GPU参照2prefixも含む全6runを直列実行した。captureは精度検査用で、ここでの実行時間を速度として使わない。

| backend | cluster | 保存PF state | QC fast | ambiguity update | integer fast scan | initx fast |
|---|---:|---:|---:|---:|---:|---:|
| cpu | 0 | 200 | 800 | 400 | 1200 | 5628 |
| gpu | 0 | 200 | 800 | 400 | 1200 | 5628 |
| cpu | 1 | 200 | 800 | 400 | 1200 | 5626 |
| gpu | 1 | 200 | 800 | 400 | 1200 | 5626 |

全候補でQC/scans/initxのfallbackは0。phase2の全1600QC call・全800ambiguity updateについて、全入力/出力の型・順序・float/array bits・layout・可変object参照関係を照合した。ambiguity update前後の全Obs fieldsも不変。CPU/GPUの既存の丸め差を相互に正規化せず、同backendの元値と比較する。PFの全保存state/RNG、navigation x/P、全軌跡arrayと従来metadataが一致した。これは1route/seedのprefixで、全route/多seed精度を一般保証するものではない。

## 全RTK processの制御比較

元D236CPU、組合せ、元CPU A/Aをcluster0の同400epochで比較する。D268で検証済みの完全RTK timing bodyを変更せず、factory bindingだけを差し替える。元PF/観測/IMU/FGO/AR、両guard/一時配列/compiled QC/整数管理/一括storeを全てprocess timerに含める。入力decode/constructor/source検証/出力保存は外。診断state/deepcopy/JSON/gzipだけを外す元AST undoも保持する。

3完全prefixをwarm、その後5round×3variantを順序回転し、全18prefix/7200RTKepoch/3600PFcallの軌跡と設定が元に一致した。初期の遅いepochや不利なroundも集計に残す。事前gateは全phase2の200epoch合計で中央値5%以上/4of5改善/A/A絶対中央値5%未満、全400epochでは3%以上/4of5改善/A/A絶対中央値5%未満。丸め前で判定する。許可CPU1本/BLAS1/通常優先度/非占有、全raw/対照を保存し、完了した計測は再試行しない。

| round | phase2短縮 | phase2 A/A | 全400短縮 | 全400 A/A |
|---|---:|---:|---:|---:|
| 1 | +30.137% | +30.671% | +34.763% | +32.937% |
| 2 | +50.549% | +12.216% | +47.507% | +18.972% |
| 3 | +18.398% | +2.752% | +20.478% | +7.073% |
| 4 | +3.150% | -11.375% | +7.051% | -7.745% |
| 5 | +27.611% | +17.426% | +17.982% | +9.879% |

phase2中央値+27.611%（5/5改善）、A/A +12.216%。全400中央値+20.478%（5/5改善）、A/A +9.879%。このnativeCPU/cluster0 prefixの事前速度gateはFAIL。全体の効果が未確定のため採用しない。 A/A対照条件は絶対中央値5%未満であり、対照の大きな時間差を候補の効果へ含めない。GPU backendとcluster1の全RTK速度、application起動/I/Oやfullroute/multiseedの速度は未確立。製品/既定は変更しない。

## 保存と適用範囲

36製品source・5installed PE・PF build PE・98既報・48archive/8899filesを保全する。元D247/D268のsourceと失敗を保持し、新しいsource/実走capture/全raw/独立auditをarchiveへ保存する。新しいCUDA sanitizerは実行していない。Numba/Python一時配列は既存の両実装と同じで、heap peakは未測定。既存scalar NaN snapshot codecの制限を引き継ぐ。任意fork/API、実行時の定数/compiled-helperメモリ改変、同じ状態への並行書込みやOOMの一般保証はない。

[結果](../benchmarks/rbpf_preprocess_bundle_20261017.json)、[完了照合](../benchmarks/rbpf_preprocess_bundle_completion_check_20261017.json)、[factory](../benchmarks/rbpf_preprocess_bundle.py)、[実走capture](../benchmarks/bench_rbpf_preprocess_bundle_field.py)、[全RTK比較](../benchmarks/bench_rbpf_preprocess_bundle_route.py)、[相互作用試験](../tests/test_rbpf_preprocess_bundle.py)。構成元の[D247](rbpf_ambiguity_bookkeeping_2026_10_16.md)と[D268](rbpf_qcedit_batch_2026_10_17.md)の判定は変更していない。
