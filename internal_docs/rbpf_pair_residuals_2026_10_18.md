# D285: 初回有効PF更新の待ちをnative境界へ絞る (2026-10-18)

D284の保存済み4800RTK時計では、全5組で最初の有効PF更新epoch203の先行側が約1.1秒長かった。新しい2prefix/800RTK/400PFの区間診断でも、先行scoreのnative呼出しは1.167643秒、直後の後行scoreは3.2343msだった。400保存状態、2最終状態、2軌跡は従来基準と一致し、31tests PASS。高速化を採用した結果ではなく、次の調査対象を具体化した結果である。

## 完了済みの時計を全て保って調べる

同じD236のA/AAをepoch・pairごとに先行反転する元eventsを使い、A−AAから先行−後行へ向きをそろえる。旧metadataの最初3skip/後197batchからepoch203を照合し、全400epoch×2arm×6pairの順序と生整数nsを検査する。元wall/process/thread CPU、warm、遅いepochを全て残す。

| 保存primary組 | epoch203の先行−後行wall ms | 同thread CPU ms |
|---|---:|---:|
| 1 | 1095.446 | -15.625 |
| 2 | 1126.307 | +15.625 |
| 3 | 1082.368 | -15.625 |
| 4 | 1183.471 | +15.625 |
| 5 | 1187.010 | +15.625 |

先行−後行wallの中央値は1.126307秒。各epoch・phaseで`wall差 = thread CPU差 + (wall−thread CPU)差`へ整数nsで正確に足し戻す。後者は待機/割込/時計量子化などを含み、原因をGPU、OS、GCへ断定しない。Windows CPU時計の量子化や負の差をclipしない。

後半全200epochのthread CPU絶対A/A差中央値も6.329%ある。最初の待ちだけが全ばらつきの原因とは言えず、そのepochを引いて旧gateを再判定しない。D284の校正FAIL、D282の速度FAILはそのまま維持する。

## 新しいnative区間診断

凍結D284の独立driver protocolと、以前検証したhost span markerを組み合わせ、1組だけ両400epochを最後まで実行した。元のmainをFunctionTypeの明示的な呼出し側bindingで選択し、2つの独立RTK/観測/nav/graph/RNGと同じ同期D236 scorerを使う。既存scoreの元ステートメントをそのまま実行し、9区間と外側wall/process/thread CPUを記録する。数値source/module globalsを書き換えない。

| epoch203 | 完全RTK ms | score ms | native ms | score thread CPU標本 ms |
|---|---:|---:|---:|---:|
| AA | 2536.133 | 1169.088 | 1167.643 | 0.000 |
| A | 136.715 | 4.521 | 3.234 | 0.000 |

nativeは`self.dll.rbpf_score(...)`の前から、直後のstatus検査前まで。ctypes/C API内部のhost pack、H2D、kernel launch/計算、完了待ち、D2H/返却を含み、純粋なkernel実行時間ではない。先行scoreのCPU標本0も実仕事0を意味しない。全394scoreを元RTK epochに結び、各native≤score≤対応RTKと区間和を照合した。観測はoverheadを持つため、元の無観測時計から差し引かない。

両方の200postprocess状態は全体がA/A一致。元PF-return基準とは事前宣言済みroot `rtk`境界だけを除き、粒子/出力/RNG等を完全一致で確認する。2最終状態は同root boundaryを含め旧post-driver goldへ一致し、2軌跡の全114配列はdtype/shape/bytes一致。cfg/input/外部tc・cssrlib/GTSAM PEと全従来metadataも照合した。粒子/候補/FP64/prior/IRLS/threshold/FIX/feedback/RNGの変更はない。

## APIトレース付きの短いidle対照

保存済み実score入力を1つ、同じworkspaceで初回・直後・15秒idle後・その直後の4回だけ実行し、全返却bitと入力不変を旧GPU参照へ照合した。Nsight下の別条件であり、フルRTKのCPU処理やPython/NumPy設定と同一ではない。長い待ちはこの対照では再現しなかった。

| API対照 | native host ms | H2D API ms | launch API ms | D2H API ms | GPU kernel ms |
|---|---:|---:|---:|---:|---:|
| initial | 2.218 | 0.136 | 1.679 | 0.241 | 0.186 |
| immediate | 0.539 | 0.091 | 0.087 | 0.244 | 0.185 |
| after_idle15 | 57.148 | 0.782 | 0.051 | 56.224 | 1.782 |
| immediate_after_idle | 3.043 | 0.085 | 0.062 | 2.764 | 1.746 |

4runtime launch、8runtime memcpyを4kernel/8copyへcorrelationIdで全対応し、転送方向・byte数・順序を独立SQLite auditで確認した。15秒idle後にはD2H APIの待ちが見えるが、約1.1秒のRTK現象の原因をこの小さい対照から同定しない。API spanからGPU実稼働率や最適化利得も作らない。既定のpower/clock/GC policy変更、GPU keepalive、余分なwarm kernelや数値cacheは追加していない。

4数値計算とscorer close後のJSON保存は完了したが、プロファイラー終了処理が待機した。正確なrunner/agent/target argv・作成時刻・親子関係を保存し、今回のsessionだけをstop、開始からの全データを保持してnsys-rep/SQLiteをexportした。最初のcloserはtarget終了を要求するguardで停止し、変更0。別closerで完了済み結果とexportを確認して、再照合した今回のtarget/agent/CLIだけを終了させた。他processには触れていない。launcher exit0だけを正常終了の証明にせず、この経緯と原本を全て残す。数値測定の取り直しはない。

## 検証と保全

31unique testsは全clock/event順序・欠落・bool/float/負値・phase/完了・初回役割反転・signed residual保存と整数保存則、score引数/結果/例外の同一性、再帰/無所有context拒否、stale/破損/長過ぎるtraceの記録を検査した。新しいGPU/native buildやsanitizer実行はなく、修正済みD236 DLLと既存host markerを再利用する。既存87profile testsは歴史証拠であり今回の31件へ加算しない。

別auditで全4800保存時計/2400組差、800新RTK時計/394score trace、400状態/2最終状態/2軌跡とAPI対照4callを照合した。130過去reports・64archives/14725files・36製品sources・5installed PE・PF buildを保全する。製品/既定/精度許容差は維持。fullroute/多seed精度・連続RTK速度・GPU使用率・新しい高速化の採用は未確立。

[結果](../benchmarks/rbpf_pair_residuals_20261018.json)、[完了照合](../benchmarks/rbpf_pair_residuals_completion_check_20261018.json)、[解析/observer](../benchmarks/rbpf_pair_residuals.py)、[tests](../tests/test_rbpf_pair_residuals.py)。
