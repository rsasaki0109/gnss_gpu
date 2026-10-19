# D259: 全score呼び出しのホスト区間計測 (2026-10-17)

GPU内部の小さな変更が全scoreでは安定した上積みにならなかったため、GPU呼び出し前の準備を分けて測った。今回の成果は次の改善箇所を選ぶための診断であり、新たな高速化の採用やGPU移行ではない。製品・既定・測位モデルは変更していない。

## 比較と較正

全394実走保存入力（cluster_relin0/1各197）でoriginal、同じcode objectのclone、profile、original_AA、profile_AAを交互・反転順に実行。5round×3groupで29550timed/1970warm。各roundは197件のinteger-ns合計のpaired比を3group中央値にし、結果は5round中央値。事前基準は下記5対比すべて絶対中央値5%未満、両条件で通過。完了した計測のやり直しはない。

下表は全scoreの計測上乗せ（正は遅化）。負の値は高速化の証明ではなく、揺らぎを含む対照値である。全round値を結果JSONに残した。

| 対比 | c0 | c1 |
|---|---:|---:|
| profile_vs_original | -0.400% | +0.492% |
| profile_vs_clone | +1.103% | +0.134% |
| clone_control | +0.464% | +0.212% |
| original_AA | -0.784% | +0.411% |
| profile_AA | -0.104% | +0.423% |

全score時計には従来dedup/pack/ctypes/C API/転送/計算/完了待ち/返却とtrace tupleの公開までを含む。状態復元、setup、warm、hash、trace検証・圧縮保存は時計外。BLAS1・許可CPU1個・通常優先度・非占有環境。異なる実験の絶対時間とは比較しない。

## 区間の診断値

profile/profile_AAのtimed各5910call/条件について、各callの区間nsを記録totalで割り、その割合の中央値を示す。各項目を独立に中央値にするため、合計の厳密な100%は要求しない。記録totalの中央値はc0/c1で2.879850/3.596300ms。

| 区間 | c0 | c1 |
|---|---:|---:|
| 入口 | 0.044% | 0.035% |
| 仮説のキー生成・重複除去 | 8.933% | 6.640% |
| 型・次元・groupの確認 | 1.039% | 0.815% |
| 観測・meta・pair配列の準備 | 2.042% | 1.614% |
| 全仮説の整数DD展開 | 10.828% | 8.029% |
| 状態検索・centre/prior配列の準備 | 1.590% | 22.905% |
| config・安全性確認・返却配列確保 | 3.174% | 2.160% |
| ctypes/C API・転送・計算・完了待ち・読戻し | 71.024% | 56.481% |
| status確認・scatter・返却 | 0.875% | 0.696% |

ここでnativeはctypes引数検証、C API内のhost packing、host/device転送、kernel、完了待ち、読戻しをまとめたwall spanである。純粋なGPU演算時間やkernel時間ではない。他の区間にもcallback、marker、スケジューリング・待ちが入る。GPU使用率・全PFのGPU化率・全PF/FGO高速化率へ読み替えない。

c1の状態検索・centre/prior準備は約23%、DD展開は約8%、dedupは約7%。c0の状態準備は約1.6%、DD展開は約11%、dedupは約9%。この差から、cluster有効時の状態検索を次の優先候補とする。元の_cluster_stateは仮説ごとにsupport/canonical keyを確認し、exact hit以外では各親状態との整数DD比較を呼び出して閾値・順位・同点判定を行う。この一括化には、任意精度整数、元のPython除算、閾値、同点時のNone、状態objectの同一性、custom callback/例外/変更時fallbackを保つ必要がある。

この計測だけでは重複計算を何%省けるか、全PFが何%速くなるかは確定しない。D232のDD展開やD234の親比較は過去の全PF screenが安定通過しておらず、今回の割合を理由に再採用しない。状態検索一括化も未実装・未測定。native区間の内訳と全PFへの寄与は別途確認が必要である。

## 正確性と保存

87tests PASS（既存pytest設定warning1件）。54数値fixtureのCPU/GPU score・135診断値・status・入力bits、境界/empty/unsupported/unsafe/error/warning、返却配列ownership、連続生成/closeを確認。3seed×4設定×4epochのPF比較では全保存状態・RNG・graph eventが一致した。元ASTからmarker/trace/finallyを除くと元scoreに一致し、cloneは元code objectをそのまま使用。profileは元のlive numeric globalsを使い、補助関数2個はclosure cellで渡してmoduleを書き換えない。live global変更と途中例外でも元の動作・trace解放を検査した。数値許容差を追加していない。

全1970warmで凍結GPU参照のscore・status・全135診断値の一致と入力不変を確認し、診断hashもrawへ保存。全29550timedの返却hashが同じ参照に一致し、scalar fallback0。rawには全788warm/11820timedの10境界eventを保存し、順序・単調性・partition和・全score時計以内・paired集計を独立auditで再照合した。

最初の実装はimport時のclone bytecode一致assertでcollection停止。元class/module contextと単独AST compileでLOAD_GLOBAL/LOAD_ATTRのcall形式が異なることを診断し、元のsourceを残して別v2でcloneに元code objectを用いた。初回数値testは86PASS/1FAILで、unsafe時に両版が同じmath-domain ValueErrorを出すのにtestが例外tupleを配列として比較していた。別のchecked testで同じ例外を比較し直した。v2実装は最初の数値test以降変更していない。失敗testはbyte同一でtests/fixturesの非test名へ移し、旧pathとの対応・元plan/logを保存した。これらは数値一致や速度のFAILを隠すための再計測ではない。

新たなCUDA source/binaryや新たなmemcheck/racecheck/synccheck実行はない。凍結D253同期修正版DLLをそのまま使用した。新たな400状態field走行、fullroute/multiseed測位精度評価、allocation/async障害注入も行っていない。既存NumPy/nativeの演算差とscalar NaN codecの限界を引き継ぐ。今回のビット一致は上記入力と試験範囲の非劣化確認である。

3contextの既存numeric stagingはhost/deviceそれぞれ3430056B（各1143352B）。追加はcall-local最大10eventと最後のimmutable tuple、Python heap peakは未計測。新しいdevice bufferやcross-call/epoch cacheはない。観測・仮説・粒子・prior・FP64・演算順・IRLS回数・閾値・RNGを維持した。36製品source・5installed PE・PF build PE・78過去reports・38archives6331filesを照合した。

[結果](../benchmarks/rbpf_score_host_phases_20261017.json)、[完了照合](../benchmarks/rbpf_score_host_phases_completion_check_20261017.json)、[計測実装](../benchmarks/rbpf_score_host_phases_v2.py)、[driver](../benchmarks/bench_rbpf_score_host_phases.py)、[checked tests](../tests/test_rbpf_score_host_phases_checked.py)。
