# D286: プロファイラーなしで長いH2D区間を観測 (2026-10-18)

通常の実RTKでepoch203の先行score native呼出し1112.153msを再現し、うちH2Dを含むhost区間が1073.550msだった。後行nativeは2.891ms。同じfield Python/NumPyの別4call対照でも、15秒idle後はnative903.280ms、H2D区間868.494msを観測した。待ちが再発する条件を絞る診断であり、速度改善の採用結果ではない。

38unique tests PASS。両clusterの保存済み実score入力394件で元DLL/新DLLの無観測export/観測exportの値・status・診断配列をdtype/shape/bytesで一致確認した。実RTKも400postprocess状態・2最終状態・2軌跡全114配列が従来基準と一致した。粒子数/候補/FP64/IRLS/prior/FIX/feedback/RNG/許容差の変更はない。

## 観測版の精度と独立性

修正済みD236の元translation unit全21000bytesを新sourceのprefixとして保持する。既存rbpf_score/exportと全device関数はそのまま。別export rbpf_score_traceは元score全関数の複製にhost-only QPC/GetThreadTimesマーカー6個とRAII入口/出口scopeを加え、7文の除去と関数名の復元で元関数全bytesへ戻る。register_lu.cuhも元headerとbyte一致。新旧DLLの全3device関数はSASS命令/operand/encoding wordsとregister/stack/shared/local等が完全一致した。

CUDA API/event/sync/kernelは追加しない。Windows calling-thread TLSにowner/generation/frequency/count/flagsとraw QPC/user100ns/kernel100nsを保存し、同じthread・ownerの完了済みtraceだけを返す。無観測exportは明示的な対照として残す。Pythonは元HostPhaseScorerの数値処理を使い、単独instanceの呼出し先だけを選択する。戻り値の数値objectは加工しない。

最初のbuildは既存headerの参照漏れで停止した。初回3source・generator・proof・plan/log/失敗結果を保存し、元headerをそのまま加えたdependency closureを別proofで凍結し、別build directoryで成功した。凍結済み数値関数の修正や、数値検証失敗の取り直しはない。

## 通常RTKの全区間を残した診断

外部プロファイラーを使わず、独立した2driverの400epochを従来の先行反転protocolで最後まで実行した。元D284/D285の数値source/main/protocolと通常GC/CPU1/BLAS1を使用し、1組が完了した時点で終了する。旧5組の校正を再測定しない。全800RTK/400PF/394scoreを保持する。

| epoch203 | native host ms | H2D包含区間 ms | launch包含区間 ms | D2H包含区間 ms |
|---|---:|---:|---:|---:|
| AA | 1112.1532 | 1073.5500 | 36.3948 | 2.0579 |
| A | 2.8910 | 0.0525 | 0.0198 | 2.7591 |

H2D区間はcudaMemcpyHostToDevice直前から、戻り値確認・出力サイズ計算後のkernel分岐直前まで。D2H区間はcudaMemcpyDeviceToHost直前から完了・戻り値確認後のhost結果copy直前まで。各区間はruntime処理/待機/host命令/割込/マーカーoverheadを含む。データ転送だけの物理的な時間や純粋なkernel実行時間とは呼ばない。H2D区間の長さだけからdriver/power/OS/メモリーの原因を決めない。

各traceの元38uint64wordsを残す。QPCを早期にnsへ丸めず、7区間の整数tick/user/kernelの和が全体へ正確に戻ること、native QPC duration <= Python native <= score <= 対応RTKを整数有理数で確認した。CPU標本0や15.625ms刻みもそのまま保持し、実仕事0とは解釈しない。warm/cold/遅いepochを除かず、時計からoverheadも差し引かない。

両armの200状態全体がA/A完全一致。元PF-return基準との比較では事前宣言済みroot rtk境界だけを除き、両最終状態はその境界も含め旧post-driver goldへ一致する。cfg/metadata/input/外部tc/cssrlib/GTSAM PEを保全し、軌跡2本の全114配列はdtype/shape/bytes一致した。2prefix/400epochの検証範囲であり、full route/多seedの新しい精度評価ではない。

## 同じfield環境の15秒idle対照

fieldと同じPython3.12.10/NumPy2.4.6を使い、保存済みcluster0の先頭実score入力を、同一workspaceで初回・直後・15秒idle後・その直後の4回だけ実行した。全4callの入力不変・scalar fallback0・結果digestの旧GPU参照一致を確認した。D285のNsight対照とプロファイラー/NumPy条件をそろえた比較ではないため、差をツールの原因と断定しない。

| 対照call | native host ms | H2D包含区間 ms | launch包含区間 ms | D2H包含区間 ms |
|---|---:|---:|---:|---:|
| initial | 1.3658 | 0.0644 | 0.9158 | 0.2879 |
| immediate | 0.4499 | 0.0466 | 0.0180 | 0.3282 |
| after_idle15 | 903.2798 | 868.4936 | 0.0373 | 34.6729 |
| immediate_after_idle | 2.1193 | 0.0596 | 0.0309 | 1.9589 |

初回の短いcallの後にもidleで長いH2D区間が生じた。最初のkernel読込み一回だけで説明できるとは確定していない。次は既存pageable staging/転送経路とidleからの復帰条件を、同じ精度基準と計測条件で切り分ける。ここでpower policy変更、keepalive、余分なwarm kernel、推定処理の間引きは導入しない。

局所的な約1秒の待ちを同定できても、全RTKの改善率は未計測である。旧D284の校正FAIL、D282の速度FAILを保持し、epoch除外や別CPU gateで再判定しない。後半全epochのthreadCPU絶対A/A差中央値6.329percentも未解決であり、今回の待ちが全ばらつきの原因とは言えない。

## 検証と保全

38件はraw traceの型/長さ/header/phase/単調性/整数保存、部分CPU/エラー退出、元/新無観測/新観測のCPU/GPU数値bit、両cluster全394実入力、owner/capacity/別thread/空入力時のstale拒否、invalid native callの戻り値と出力保持を検査する。394ループを394testsへ加算しない。新device処理はなく全3kernel機械語一致のため、今回は新たなCompute Sanitizer再実行を行っていない。

独立auditはhelperをimportせずsource全関数復元、SASSとresource、394+4traceのraw整数保存/境界、全800RTK順序、状態/最終状態/軌跡を再照合した。132過去reports・65archives/15185files・36製品source・5installed PE・PF buildを保全する。製品/既定/数値許容差/seedを維持し、新しい高速化・GPU使用率・driver根因の主張は行わない。

[結果](../benchmarks/rbpf_native_call_trace_20261018.json)、[完了照合](../benchmarks/rbpf_native_call_trace_completion_check_20261018.json)、[観測wrapper](../benchmarks/rbpf_native_call_trace.py)、[native source](../benchmarks/probes/rbpf_native_call_trace/probe.cu)、[tests](../tests/test_rbpf_native_call_trace.py)。
